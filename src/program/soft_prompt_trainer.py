import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import EvalPrediction, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.deepspeed import deepspeed_init
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import (
   IterableDatasetShard,
   find_batch_size,
   nested_concat,
   nested_detach,
   nested_numpify,
   nested_truncate,
)
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, has_length
from transformers.utils import is_sagemaker_mp_enabled, is_torch_tpu_available, logging


logger = logging.get_logger(__name__)




if is_sagemaker_mp_enabled():
   import smdistributed.modelparallel.torch as smp
   from smdistributed.modelparallel import __version__ as SMP_VERSION


   IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")


   from transformers.trainer_pt_utils import (
       smp_forward_backward,
       smp_forward_only,
       smp_gather,
       smp_nested_concat,
   )
else:
   IS_SAGEMAKER_MP_POST_1_10 = False




class SoftPromptTrainer(Trainer):
   def __init__(
       self,
       model: Union[PreTrainedModel, nn.Module] = None,
       args: TrainingArguments = None,
       data_collator: Optional[DataCollator] = None,
       train_dataset: Optional[Dataset] = None,
       eval_dataset: Optional[Dataset] = None,
       tokenizer: Optional[PreTrainedTokenizerBase] = None,
       model_init: Callable[[], PreTrainedModel] = None,
       compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
       callbacks: Optional[List[TrainerCallback]] = None,
       optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
           None,
           None,
       ),
       save_file_name: str = "soft_prompt",
   ):
       super().__init__(
           model=model,
           args=args,
           data_collator=data_collator,
           train_dataset=train_dataset,
           eval_dataset=eval_dataset,
           tokenizer=tokenizer,
           model_init=model_init,
           compute_metrics=compute_metrics,
           callbacks=callbacks,
           optimizers=optimizers,
       )
       self.save_file_name = save_file_name


   def _save(self, output_dir: Optional[str] = None, state_dict=None):
       # If we are executing this function, we are the process zero, so we don't check for that.
       output_dir = output_dir if output_dir is not None else self.args.output_dir
       os.makedirs(output_dir, exist_ok=True)
       # logger.info(f"Saving model checkpoint to {output_dir}")
       # # Save a trained model and configuration using `save_pretrained()`.
       # # They can then be reloaded using `from_pretrained()`
       # if not isinstance(self.model, PreTrainedModel):
       #     if isinstance(unwrap_model(self.model), PreTrainedModel):
       #         if state_dict is None:
       #             state_dict = self.model.state_dict()
       #         unwrap_model(self.model).save_pretrained(
       #             output_dir, state_dict=state_dict
       #         )
       #     else:
       #         logger.info(
       #             "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
       #         )
       #         if state_dict is None:
       #             state_dict = self.model.state_dict()
       #         torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
       # else:
       #     self.model.save_pretrained(output_dir, state_dict=state_dict)
       # if self.tokenizer is not None:
       #     self.tokenizer.save_pretrained(output_dir)


       # # Good practice: save your training arguments together with the trained model
       self.model.save_soft_prompt(path=output_dir, filename=f"{self.save_file_name}.pt")


   def evaluation_loop(
       self,
       dataloader: DataLoader,
       description: str,
       prediction_loss_only: Optional[bool] = None,
       ignore_keys: Optional[List[str]] = None,
       metric_key_prefix: str = "eval",
   ) -> EvalLoopOutput:
       """
       Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.


       Works both with or without labels.
       """
       args = self.args


       prediction_loss_only = (
           prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
       )


       # if eval is called w/o train init deepspeed here
       if args.deepspeed and not self.deepspeed:


           # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
           # from the checkpoint eventually
           deepspeed_engine, _, _ = deepspeed_init(
               self, num_training_steps=0, resume_from_checkpoint=None, inference=True
           )
           self.model = deepspeed_engine.module
           self.model_wrapped = deepspeed_engine
           self.deepspeed = deepspeed_engine


       model = self._wrap_model(self.model, training=False, dataloader=dataloader)


       # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
       # while ``train`` is running, cast it to the right dtype first and then put on device
       if not self.is_in_train:
           if args.fp16_full_eval:
               model = model.to(dtype=torch.float16, device=args.device)
           elif args.bf16_full_eval:
               model = model.to(dtype=torch.bfloat16, device=args.device)


       batch_size = self.args.eval_batch_size


       logger.info(f"***** Running {description} *****")
       if has_length(dataloader):
           logger.info(f"  Num examples = {self.num_examples(dataloader)}")
       else:
           logger.info("  Num examples: Unknown")
       logger.info(f"  Batch size = {batch_size}")


       model.eval()


       self.callback_handler.eval_dataloader = dataloader
       # Do this before wrapping.
       eval_dataset = getattr(dataloader, "dataset", None)


       if is_torch_tpu_available():
           dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)


       if args.past_index >= 0:
           self._past = None


       # Initialize containers
       # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
       losses_host = None
       src_losses_host = None
       tgt_losses_host = None
       kl_losses_host = None
       preds_host = None
       labels_host = None
       inputs_host = None


       # losses/preds/labels on CPU (final containers)
       all_losses = None
       all_src_losses = None
       all_tgt_losses = None
       all_kl_losses = None
       all_losses = None
       all_preds = None
       all_labels = None
       all_inputs = None
       # Will be useful when we have an iterable dataset so don't know its length.


       observed_num_examples = 0
       # Main evaluation loop
       for step, inputs in enumerate(dataloader):
           # Update the observed num examples
           observed_batch_size = find_batch_size(inputs)
           if observed_batch_size is not None:
               observed_num_examples += observed_batch_size
               # For batch samplers, batch_size is not known by the dataloader in advance.
               if batch_size is None:
                   batch_size = observed_batch_size


           # Prediction step
           # prediciton step で src, tgt, kl ロスを返すように修正 (done)
           loss, src_loss, tgt_loss, kl_loss, logits, labels = self.prediction_step(
               model, inputs, prediction_loss_only, ignore_keys=ignore_keys
           )
           inputs_decode = (
               self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None
           )


           if is_torch_tpu_available():
               xm.mark_step()


           # Update containers on host
           # src, tgt, kl losses, losses_host を定義 (done)
           if loss is not None:
               losses = self._nested_gather(loss.repeat(batch_size))
               losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
           if src_loss is not None:
               src_losses = self._nested_gather(src_loss.repeat(batch_size))
               src_losses_host = (
                   src_losses if src_losses_host is None else torch.cat((src_losses_host, src_losses), dim=0)
               )
           if tgt_loss is not None:
               tgt_losses = self._nested_gather(tgt_loss.repeat(batch_size))
               tgt_losses_host = (
                   tgt_losses if tgt_losses_host is None else torch.cat((tgt_losses_host, tgt_losses), dim=0)
               )
           if kl_loss is not None:
               kl_losses = self._nested_gather(kl_loss.repeat(batch_size))
               kl_losses_host = (
                   kl_losses if kl_losses_host is None else torch.cat((kl_losses_host, kl_losses), dim=0)
               )
           if labels is not None:
               labels = self._pad_across_processes(labels)
               labels = self._nested_gather(labels)
               labels_host = (
                   labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
               )
           if inputs_decode is not None:
               inputs_decode = self._pad_across_processes(inputs_decode)
               inputs_decode = self._nested_gather(inputs_decode)
               inputs_host = (
                   inputs_decode
                   if inputs_host is None
                   else nested_concat(inputs_host, inputs_decode, padding_index=-100)
               )
           if logits is not None:
               logits = self._pad_across_processes(logits)
               logits = self._nested_gather(logits)
               if self.preprocess_logits_for_metrics is not None:
                   logits = self.preprocess_logits_for_metrics(logits, labels)
               preds_host = (
                   logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
               )
           self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)


           # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
           if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
               if losses_host is not None:
                   losses = nested_numpify(losses_host)
                   all_losses = (
                       losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                   )
               if preds_host is not None:
                   logits = nested_numpify(preds_host)
                   all_preds = (
                       logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                   )
               if inputs_host is not None:
                   inputs_decode = nested_numpify(inputs_host)
                   all_inputs = (
                       inputs_decode
                       if all_inputs is None
                       else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                   )
               if labels_host is not None:
                   labels = nested_numpify(labels_host)
                   all_labels = (
                       labels
                       if all_labels is None
                       else nested_concat(all_labels, labels, padding_index=-100)
                   )


               # Set back to None to begin a new accumulation
               losses_host, preds_host, inputs_host, labels_host = None, None, None, None


       if args.past_index and hasattr(self, "_past"):
           # Clean the state at the end of the evaluation loop
           delattr(self, "_past")


       # Gather all remaining tensors and put them back on the CPU
       # src, tgt, kl all_losses
       if losses_host is not None:
           losses = nested_numpify(losses_host)
           all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
       if src_losses_host is not None:
           src_losses = nested_numpify(src_losses_host)
           all_src_losses = (
               src_losses if all_src_losses is None else np.concatenate((all_src_losses, src_losses), axis=0)
           )
       if tgt_losses_host is not None:
           tgt_losses = nested_numpify(tgt_losses_host)
           all_tgt_losses = (
               tgt_losses if all_tgt_losses is None else np.concatenate((all_tgt_losses, tgt_losses), axis=0)
           )
       if kl_losses_host is not None:
           kl_losses = nested_numpify(kl_losses_host)
           all_kl_losses = (
               kl_losses if all_kl_losses is None else np.concatenate((all_kl_losses, kl_losses), axis=0)
           )
       if preds_host is not None:
           logits = nested_numpify(preds_host)
           all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
       if inputs_host is not None:
           inputs_decode = nested_numpify(inputs_host)
           all_inputs = (
               inputs_decode
               if all_inputs is None
               else nested_concat(all_inputs, inputs_decode, padding_index=-100)
           )
       if labels_host is not None:
           labels = nested_numpify(labels_host)
           all_labels = (
               labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
           )


       # Number of samples
       if has_length(eval_dataset):
           num_samples = len(eval_dataset)
       # The instance check is weird and does not actually check for the type, but whether the dataset has the right
       # methods. Therefore we need to make sure it also has the attribute.
       elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
           num_samples = eval_dataset.num_examples
       else:
           if has_length(dataloader):
               num_samples = self.num_examples(dataloader)
           else:  # both len(dataloader.dataset) and len(dataloader) fail
               num_samples = observed_num_examples
       if num_samples == 0 and observed_num_examples > 0:
           num_samples = observed_num_examples


       # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
       # samplers has been rounded to a multiple of batch_size, so we truncate.
       if all_losses is not None:
           all_losses = all_losses[:num_samples]
       if all_src_losses is not None:
           all_src_losses = all_src_losses[:num_samples]
       if all_tgt_losses is not None:
           all_tgt_losses = all_tgt_losses[:num_samples]
       if all_kl_losses is not None:
           all_kl_losses = all_kl_losses[:num_samples]
       if all_preds is not None:
           all_preds = nested_truncate(all_preds, num_samples)
       if all_labels is not None:
           all_labels = nested_truncate(all_labels, num_samples)
       if all_inputs is not None:
           all_inputs = nested_truncate(all_inputs, num_samples)


       # Metrics!
       if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
           if args.include_inputs_for_metrics:
               metrics = self.compute_metrics(
                   EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
               )
           else:
               metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
       else:
           metrics = {}


       # To be JSON-serializable, we need to remove numpy types or zero-d tensors
       metrics = denumpify_detensorize(metrics)


       if all_losses is not None:
           metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
       if all_src_losses is not None:
           metrics[f"{metric_key_prefix}_src_loss"] = all_src_losses.mean().item()
       if all_tgt_losses is not None:
           metrics[f"{metric_key_prefix}_tgt_loss"] = all_tgt_losses.mean().item()
       if all_kl_losses is not None:
           metrics[f"{metric_key_prefix}_kl_loss"] = all_kl_losses.mean().item()


       # Prefix all keys with metric_key_prefix + '_'
       for key in list(metrics.keys()):
           if not key.startswith(f"{metric_key_prefix}_"):
               metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)


       return EvalLoopOutput(
           predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples
       )


   def prediction_step(
       self,
       model: nn.Module,
       inputs: Dict[str, Union[torch.Tensor, Any]],
       prediction_loss_only: bool,
       ignore_keys: Optional[List[str]] = None,
   ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
       """
       Perform an evaluation step on `model` using `inputs`.


       Subclass and override to inject custom behavior.


       Args:
           model (`nn.Module`):
               The model to evaluate.
           inputs (`Dict[str, Union[torch.Tensor, Any]]`):
               The inputs and targets of the model.


               The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
               argument `labels`. Check your model's documentation for all accepted arguments.
           prediction_loss_only (`bool`):
               Whether or not to return the loss only.
           ignore_keys (`Lst[str]`, *optional*):
               A list of keys in the output of your model (if it is a dictionary) that should be ignored when
               gathering predictions.


       Return:
           Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
           logits and labels (each being optional).
       """
       has_labels = all(inputs.get(k) is not None for k in self.label_names)
       inputs = self._prepare_inputs(inputs)
       if ignore_keys is None:
           if hasattr(self.model, "config"):
               ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
           else:
               ignore_keys = []


       # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
       if has_labels:
           labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
           if len(labels) == 1:
               labels = labels[0]
       else:
           labels = None


       with torch.no_grad():
           if is_sagemaker_mp_enabled():
               raw_outputs = smp_forward_only(model, inputs)
               if has_labels:
                   if isinstance(raw_outputs, dict):
                       loss_mb = raw_outputs["loss"]
                       logits_mb = tuple(
                           v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"]
                       )
                   else:
                       loss_mb = raw_outputs[0]
                       logits_mb = raw_outputs[1:]


                   loss = loss_mb.reduce_mean().detach().cpu()
                   logits = smp_nested_concat(logits_mb)
               else:
                   loss = None
                   if isinstance(raw_outputs, dict):
                       logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                   else:
                       logits_mb = raw_outputs
                   logits = smp_nested_concat(logits_mb)
           else:
               if has_labels:
                   with self.compute_loss_context_manager():
                       loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                   loss = loss.mean().detach()


                   if isinstance(outputs, dict):
                       logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                   else:
                       logits = outputs[1:]
               else:
                   loss = None
                   with self.compute_loss_context_manager():
                       outputs = model(**inputs)
                   if isinstance(outputs, dict):
                       logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                   else:
                       logits = outputs
                   # TODO: this needs to be fixed and made cleaner later.
                   if self.args.past_index >= 0:
                       self._past = outputs[self.args.past_index - 1]


       if prediction_loss_only:
           return outputs.loss, outputs.src_plm_loss, outputs.tgt_plm_loss, outputs.kl_loss, None, None


       logits = nested_detach(logits)
       if len(logits) == 1:
           logits = logits[0]


       return (loss, logits, labels)


   def compute_loss(self, model, inputs, return_outputs=False):
       """
       How the loss is computed by Trainer. By default, all models return the loss in the first element.
       Subclass and override for custom behavior.
       """
       if self.label_smoother is not None and "labels" in inputs:
           labels = inputs.pop("labels")
       else:
           labels = None
       outputs = model(**inputs)
       # Save past state if it exists
       # TODO: this needs to be fixed and made cleaner later.
       if self.args.past_index >= 0:
           self._past = outputs[self.args.past_index]


       if labels is not None:
           if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
               loss = self.label_smoother(outputs, labels, shift_labels=True)
           else:
               loss = self.label_smoother(outputs, labels)
       else:
           if isinstance(outputs, dict) and "loss" not in outputs:
               raise ValueError(
                   "The model did not return a loss from the inputs, only the following keys: "
                   f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
               )
           # We don't use .loss here since the model may return tuples instead of ModelOutput.
           loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]


       return (loss, outputs) if return_outputs else loss




class PEFTTrainer(Trainer):
   def __init__(
       self,
       model: Union[PreTrainedModel, nn.Module] = None,
       args: TrainingArguments = None,
       data_collator: Optional[DataCollator] = None,
       train_dataset: Optional[Dataset] = None,
       eval_dataset: Optional[Dataset] = None,
       tokenizer: Optional[PreTrainedTokenizerBase] = None,
       model_init: Callable[[], PreTrainedModel] = None,
       compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
       callbacks: Optional[List[TrainerCallback]] = None,
       optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
           None,
           None,
       ),
   ):
       super().__init__(
           model=model,
           args=args,
           data_collator=data_collator,
           train_dataset=train_dataset,
           eval_dataset=eval_dataset,
           tokenizer=tokenizer,
           model_init=model_init,
           compute_metrics=compute_metrics,
           callbacks=callbacks,
           optimizers=optimizers,
       )


   def _save(self, output_dir: Optional[str] = None, state_dict=None):
       # If we are executing this function, we are the process zero, so we don't check for that.
       output_dir = output_dir if output_dir is not None else self.args.output_dir
       os.makedirs(output_dir, exist_ok=True)
       logger.info(f"Saving model checkpoint to {output_dir}")
       # # Save a trained model and configuration using `save_pretrained()`.
       # # They can then be reloaded using `from_pretrained()`
       # if not isinstance(self.model, PreTrainedModel):
       #     if isinstance(unwrap_model(self.model), PreTrainedModel):
       #         if state_dict is None:
       #             state_dict = self.model.state_dict()
       #         unwrap_model(self.model).save_pretrained(
       #             output_dir, state_dict=state_dict
       #         )
       #     else:
       #         logger.info(
       #             "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
       #         )
       #         if state_dict is None:
       #             state_dict = self.model.state_dict()
       #         torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
       # else:
       #     self.model.save_pretrained(output_dir, state_dict=state_dict)
       # if self.tokenizer is not None:
       #     self.tokenizer.save_pretrained(output_dir)


       # # Good practice: save your training arguments together with the trained model
       self.model.save_pretrained(output_dir)
   def evaluation_loop(
       self,
       dataloader: DataLoader,
       description: str,
       prediction_loss_only: Optional[bool] = None,
       ignore_keys: Optional[List[str]] = None,
       metric_key_prefix: str = "eval",
   ) -> EvalLoopOutput:
       """
       Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
       Works both with or without labels.
       """
       args = self.args


       prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only


       # if eval is called w/o train init deepspeed here
       if args.deepspeed and not self.deepspeed:
           # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
           # from the checkpoint eventually
           deepspeed_engine, _, _ = deepspeed_init(
               self, num_training_steps=0, resume_from_checkpoint=None, inference=True
           )
           self.model = deepspeed_engine.module
           self.model_wrapped = deepspeed_engine
           self.deepspeed = deepspeed_engine


       model = self._wrap_model(self.model, training=False, dataloader=dataloader)


       # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
       # while ``train`` is running, cast it to the right dtype first and then put on device
       if not self.is_in_train:
           if args.fp16_full_eval:
               model = model.to(dtype=torch.float16, device=args.device)
           elif args.bf16_full_eval:
               model = model.to(dtype=torch.bfloat16, device=args.device)


       batch_size = self.args.eval_batch_size


       logger.info(f"***** Running {description} *****")
       if has_length(dataloader):
           logger.info(f"  Num examples = {self.num_examples(dataloader)}")
       else:
           logger.info("  Num examples: Unknown")
       logger.info(f"  Batch size = {batch_size}")


       model.eval()


       self.callback_handler.eval_dataloader = dataloader
       # Do this before wrapping.
       eval_dataset = getattr(dataloader, "dataset", None)


       if is_torch_tpu_available():
           dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)


       if args.past_index >= 0:
           self._past = None


       # Initialize containers
       # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
       losses_host = None
       preds_host = None
       labels_host = None
       inputs_host = None


       # losses/preds/labels on CPU (final containers)
       all_losses = None
       all_preds = None
       all_labels = None
       all_inputs = None
       # Will be useful when we have an iterable dataset so don't know its length.


       observed_num_examples = 0
       # Main evaluation loop
       for step, inputs in enumerate(dataloader):
           # Update the observed num examples
           observed_batch_size = find_batch_size(inputs)
           if observed_batch_size is not None:
               observed_num_examples += observed_batch_size
               # For batch samplers, batch_size is not known by the dataloader in advance.
               if batch_size is None:
                   batch_size = observed_batch_size


           # Prediction step
           loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
           inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None


           if is_torch_tpu_available():
               xm.mark_step()


           # Update containers on host
           if loss is not None:
               losses = self._nested_gather(loss.repeat(batch_size))
               losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
           if labels is not None:
               labels = self._pad_across_processes(labels)
               labels = self._nested_gather(labels)
               labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
           if inputs_decode is not None:
               inputs_decode = self._pad_across_processes(inputs_decode)
               inputs_decode = self._nested_gather(inputs_decode)
               inputs_host = (
                   inputs_decode
                   if inputs_host is None
                   else nested_concat(inputs_host, inputs_decode, padding_index=-100)
               )
           if logits is not None:
               logits = self._pad_across_processes(logits)
               logits = self._nested_gather(logits)
               if self.preprocess_logits_for_metrics is not None:
                   logits = self.preprocess_logits_for_metrics(logits, labels)
               preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
           self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)


           # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
           if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
               if losses_host is not None:
                   losses = nested_numpify(losses_host)
                   all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
               if preds_host is not None:
                   logits = nested_numpify(preds_host)
                   all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
               if inputs_host is not None:
                   inputs_decode = nested_numpify(inputs_host)
                   all_inputs = (
                       inputs_decode
                       if all_inputs is None
                       else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                   )
               if labels_host is not None:
                   labels = nested_numpify(labels_host)
                   all_labels = (
                       labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                   )


               # Set back to None to begin a new accumulation
               losses_host, preds_host, inputs_host, labels_host = None, None, None, None


       if args.past_index and hasattr(self, "_past"):
           # Clean the state at the end of the evaluation loop
           delattr(self, "_past")


       # Gather all remaining tensors and put them back on the CPU
       if losses_host is not None:
           losses = nested_numpify(losses_host)
           all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
       if preds_host is not None:
           logits = nested_numpify(preds_host)
           all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
       if inputs_host is not None:
           inputs_decode = nested_numpify(inputs_host)
           all_inputs = (
               inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
           )
       if labels_host is not None:
           labels = nested_numpify(labels_host)
           all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)


       # Number of samples
       if has_length(eval_dataset):
           num_samples = len(eval_dataset)
       # The instance check is weird and does not actually check for the type, but whether the dataset has the right
       # methods. Therefore we need to make sure it also has the attribute.
       elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
           num_samples = eval_dataset.num_examples
       else:
           if has_length(dataloader):
               num_samples = self.num_examples(dataloader)
           else:  # both len(dataloader.dataset) and len(dataloader) fail
               num_samples = observed_num_examples
       if num_samples == 0 and observed_num_examples > 0:
           num_samples = observed_num_examples


       # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
       # samplers has been rounded to a multiple of batch_size, so we truncate.
       if all_losses is not None:
           all_losses = all_losses[:num_samples]
       if all_preds is not None:
           all_preds = nested_truncate(all_preds, num_samples)
       if all_labels is not None:
           all_labels = nested_truncate(all_labels, num_samples)
       if all_inputs is not None:
           all_inputs = nested_truncate(all_inputs, num_samples)


       # Metrics!
       if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
           if args.include_inputs_for_metrics:
               metrics = self.compute_metrics(
                   EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
               )
           else:
               metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
       else:
           metrics = {}


       # To be JSON-serializable, we need to remove numpy types or zero-d tensors
       metrics = denumpify_detensorize(metrics)


       if all_losses is not None:
           metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
       if hasattr(self, "jit_compilation_time"):
           metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time


       # Prefix all keys with metric_key_prefix + '_'
       for key in list(metrics.keys()):
           if not key.startswith(f"{metric_key_prefix}_"):
               metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)


       return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
   # def evaluation_loop(
   #     self,
   #     dataloader: DataLoader,
   #     description: str,
   #     prediction_loss_only: Optional[bool] = None,
   #     ignore_keys: Optional[List[str]] = None,
   #     metric_key_prefix: str = "eval",
   # ) -> EvalLoopOutput:
   #     """
   #     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.


   #     Works both with or without labels.
   #     """
   #     args = self.args


   #     prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only


   #     # if eval is called w/o train init deepspeed here
   #     if args.deepspeed and not self.deepspeed:


   #         # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
   #         # from the checkpoint eventually
   #         deepspeed_engine, _, _ = deepspeed_init(
   #             self, num_training_steps=0, resume_from_checkpoint=None, inference=True
   #         )
   #         self.model = deepspeed_engine.module
   #         self.model_wrapped = deepspeed_engine
   #         self.deepspeed = deepspeed_engine


   #     model = self._wrap_model(self.model, training=False, dataloader=dataloader)


   #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
   #     # while ``train`` is running, cast it to the right dtype first and then put on device
   #     if not self.is_in_train:
   #         if args.fp16_full_eval:
   #             model = model.to(dtype=torch.float16, device=args.device)
   #         elif args.bf16_full_eval:
   #             model = model.to(dtype=torch.bfloat16, device=args.device)


   #     batch_size = self.args.eval_batch_size


   #     logger.info(f"***** Running {description} *****")
   #     if has_length(dataloader):
   #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
   #     else:
   #         logger.info("  Num examples: Unknown")
   #     logger.info(f"  Batch size = {batch_size}")


   #     model.eval()


   #     self.callback_handler.eval_dataloader = dataloader
   #     # Do this before wrapping.
   #     eval_dataset = getattr(dataloader, "dataset", None)


   #     if is_torch_tpu_available():
   #         dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)


   #     if args.past_index >= 0:
   #         self._past = None


   #     # Initialize containers
   #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
   #     losses_host = None
   #     src_losses_host = None
   #     tgt_losses_host = None
   #     kl_losses_host = None
   #     preds_host = None
   #     labels_host = None
   #     inputs_host = None


   #     # losses/preds/labels on CPU (final containers)
   #     all_losses = None
   #     all_src_losses = None
   #     all_tgt_losses = None
   #     all_kl_losses = None
   #     all_losses = None
   #     all_preds = None
   #     all_labels = None
   #     all_inputs = None
   #     # Will be useful when we have an iterable dataset so don't know its length.


   #     observed_num_examples = 0
   #     # Main evaluation loop
   #     for step, inputs in enumerate(dataloader):
   #         # Update the observed num examples
   #         observed_batch_size = find_batch_size(inputs)
   #         if observed_batch_size is not None:
   #             observed_num_examples += observed_batch_size
   #             # For batch samplers, batch_size is not known by the dataloader in advance.
   #             if batch_size is None:
   #                 batch_size = observed_batch_size


   #         # Prediction step
   #         # prediciton step で src, tgt, kl ロスを返すように修正 (done)
   #         loss, src_loss, tgt_loss, kl_loss, logits, labels = self.prediction_step(
   #             model, inputs, prediction_loss_only, ignore_keys=ignore_keys
   #         )
   #         inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None


   #         if is_torch_tpu_available():
   #             xm.mark_step()


   #         # Update containers on host
   #         # src, tgt, kl losses, losses_host を定義 (done)
   #         if loss is not None:
   #             losses = self._nested_gather(loss.repeat(batch_size))
   #             losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
   #         if src_loss is not None:
   #             src_losses = self._nested_gather(src_loss.repeat(batch_size))
   #             src_losses_host = (
   #                 src_losses if src_losses_host is None else torch.cat((src_losses_host, src_losses), dim=0)
   #             )
   #         if tgt_loss is not None:
   #             tgt_losses = self._nested_gather(tgt_loss.repeat(batch_size))
   #             tgt_losses_host = (
   #                 tgt_losses if tgt_losses_host is None else torch.cat((tgt_losses_host, tgt_losses), dim=0)
   #             )
   #         if kl_loss is not None:
   #             kl_losses = self._nested_gather(kl_loss.repeat(batch_size))
   #             kl_losses_host = kl_losses if kl_losses_host is None else torch.cat((kl_losses_host, kl_losses), dim=0)
   #         if labels is not None:
   #             labels = self._pad_across_processes(labels)
   #             labels = self._nested_gather(labels)
   #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
   #         if inputs_decode is not None:
   #             inputs_decode = self._pad_across_processes(inputs_decode)
   #             inputs_decode = self._nested_gather(inputs_decode)
   #             inputs_host = (
   #                 inputs_decode
   #                 if inputs_host is None
   #                 else nested_concat(inputs_host, inputs_decode, padding_index=-100)
   #             )
   #         if logits is not None:
   #             logits = self._pad_across_processes(logits)
   #             logits = self._nested_gather(logits)
   #             if self.preprocess_logits_for_metrics is not None:
   #                 logits = self.preprocess_logits_for_metrics(logits, labels)
   #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
   #         self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)


   #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
   #         if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
   #             if losses_host is not None:
   #                 losses = nested_numpify(losses_host)
   #                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
   #             if preds_host is not None:
   #                 logits = nested_numpify(preds_host)
   #                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
   #             if inputs_host is not None:
   #                 inputs_decode = nested_numpify(inputs_host)
   #                 all_inputs = (
   #                     inputs_decode
   #                     if all_inputs is None
   #                     else nested_concat(all_inputs, inputs_decode, padding_index=-100)
   #                 )
   #             if labels_host is not None:
   #                 labels = nested_numpify(labels_host)
   #                 all_labels = (
   #                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
   #                 )


   #             # Set back to None to begin a new accumulation
   #             losses_host, preds_host, inputs_host, labels_host = None, None, None, None


   #     if args.past_index and hasattr(self, "_past"):
   #         # Clean the state at the end of the evaluation loop
   #         delattr(self, "_past")


   #     # Gather all remaining tensors and put them back on the CPU
   #     # src, tgt, kl all_losses
   #     if losses_host is not None:
   #         losses = nested_numpify(losses_host)
   #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
   #     if src_losses_host is not None:
   #         src_losses = nested_numpify(src_losses_host)
   #         all_src_losses = (
   #             src_losses if all_src_losses is None else np.concatenate((all_src_losses, src_losses), axis=0)
   #         )
   #     if tgt_losses_host is not None:
   #         tgt_losses = nested_numpify(tgt_losses_host)
   #         all_tgt_losses = (
   #             tgt_losses if all_tgt_losses is None else np.concatenate((all_tgt_losses, tgt_losses), axis=0)
   #         )
   #     if kl_losses_host is not None:
   #         kl_losses = nested_numpify(kl_losses_host)
   #         all_kl_losses = kl_losses if all_kl_losses is None else np.concatenate((all_kl_losses, kl_losses), axis=0)
   #     if preds_host is not None:
   #         logits = nested_numpify(preds_host)
   #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
   #     if inputs_host is not None:
   #         inputs_decode = nested_numpify(inputs_host)
   #         all_inputs = (
   #             inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
   #         )
   #     if labels_host is not None:
   #         labels = nested_numpify(labels_host)
   #         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)


   #     # Number of samples
   #     if has_length(eval_dataset):
   #         num_samples = len(eval_dataset)
   #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
   #     # methods. Therefore we need to make sure it also has the attribute.
   #     elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
   #         num_samples = eval_dataset.num_examples
   #     else:
   #         if has_length(dataloader):
   #             num_samples = self.num_examples(dataloader)
   #         else:  # both len(dataloader.dataset) and len(dataloader) fail
   #             num_samples = observed_num_examples
   #     if num_samples == 0 and observed_num_examples > 0:
   #         num_samples = observed_num_examples


   #     # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
   #     # samplers has been rounded to a multiple of batch_size, so we truncate.
   #     if all_losses is not None:
   #         all_losses = all_losses[:num_samples]
   #     if all_src_losses is not None:
   #         all_src_losses = all_src_losses[:num_samples]
   #     if all_tgt_losses is not None:
   #         all_tgt_losses = all_tgt_losses[:num_samples]
   #     if all_kl_losses is not None:
   #         all_kl_losses = all_kl_losses[:num_samples]
   #     if all_preds is not None:
   #         all_preds = nested_truncate(all_preds, num_samples)
   #     if all_labels is not None:
   #         all_labels = nested_truncate(all_labels, num_samples)
   #     if all_inputs is not None:
   #         all_inputs = nested_truncate(all_inputs, num_samples)


   #     # Metrics!
   #     if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
   #         if args.include_inputs_for_metrics:
   #             metrics = self.compute_metrics(
   #                 EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
   #             )
   #         else:
   #             metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
   #     else:
   #         metrics = {}


   #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
   #     metrics = denumpify_detensorize(metrics)


   #     if all_losses is not None:
   #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
   #     if all_src_losses is not None:
   #         metrics[f"{metric_key_prefix}_src_loss"] = all_src_losses.mean().item()
   #     if all_tgt_losses is not None:
   #         metrics[f"{metric_key_prefix}_tgt_loss"] = all_tgt_losses.mean().item()
   #     if all_kl_losses is not None:
   #         metrics[f"{metric_key_prefix}_kl_loss"] = all_kl_losses.mean().item()


   #     # Prefix all keys with metric_key_prefix + '_'
   #     for key in list(metrics.keys()):
   #         if not key.startswith(f"{metric_key_prefix}_"):
   #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)


   #     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


   # def prediction_step(
   #     self,
   #     model: nn.Module,
   #     inputs: Dict[str, Union[torch.Tensor, Any]],
   #     prediction_loss_only: bool,
   #     ignore_keys: Optional[List[str]] = None,
   # ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
   #     """
   #     Perform an evaluation step on `model` using `inputs`.


   #     Subclass and override to inject custom behavior.


   #     Args:
   #         model (`nn.Module`):
   #             The model to evaluate.
   #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
   #             The inputs and targets of the model.


   #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
   #             argument `labels`. Check your model's documentation for all accepted arguments.
   #         prediction_loss_only (`bool`):
   #             Whether or not to return the loss only.
   #         ignore_keys (`Lst[str]`, *optional*):
   #             A list of keys in the output of your model (if it is a dictionary) that should be ignored when
   #             gathering predictions.


   #     Return:
   #         Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
   #         logits and labels (each being optional).
   #     """
   #     has_labels = all(inputs.get(k) is not None for k in self.label_names)
   #     inputs = self._prepare_inputs(inputs)
   #     if ignore_keys is None:
   #         if hasattr(self.model, "config"):
   #             ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
   #         else:
   #             ignore_keys = []


   #     # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
   #     if has_labels:
   #         labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
   #         if len(labels) == 1:
   #             labels = labels[0]
   #     else:
   #         labels = None


   #     with torch.no_grad():
   #         if is_sagemaker_mp_enabled():
   #             raw_outputs = smp_forward_only(model, inputs)
   #             if has_labels:
   #                 if isinstance(raw_outputs, dict):
   #                     loss_mb = raw_outputs["loss"]
   #                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
   #                 else:
   #                     loss_mb = raw_outputs[0]
   #                     logits_mb = raw_outputs[1:]


   #                 loss = loss_mb.reduce_mean().detach().cpu()
   #                 logits = smp_nested_concat(logits_mb)
   #             else:
   #                 loss = None
   #                 if isinstance(raw_outputs, dict):
   #                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
   #                 else:
   #                     logits_mb = raw_outputs
   #                 logits = smp_nested_concat(logits_mb)
   #         else:
   #             if has_labels:
   #                 with self.compute_loss_context_manager():
   #                     loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
   #                 loss = loss.mean().detach()


   #                 if isinstance(outputs, dict):
   #                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
   #                 else:
   #                     logits = outputs[1:]
   #             else:
   #                 loss = None
   #                 with self.compute_loss_context_manager():
   #                     outputs = model(**inputs)
   #                 if isinstance(outputs, dict):
   #                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
   #                 else:
   #                     logits = outputs
   #                 # TODO: this needs to be fixed and made cleaner later.
   #                 if self.args.past_index >= 0:
   #                     self._past = outputs[self.args.past_index - 1]


   #     if prediction_loss_only:
   #         return outputs.loss, outputs.src_plm_loss, outputs.tgt_plm_loss, outputs.kl_loss, None, None


   #     logits = nested_detach(logits)
   #     if len(logits) == 1:
   #         logits = logits[0]


   #     return (loss, logits, labels)


   # def compute_loss(self, model, inputs, return_outputs=False):
   #     """
   #     How the loss is computed by Trainer. By default, all models return the loss in the first element.
   #     Subclass and override for custom behavior.
   #     """
   #     if self.label_smoother is not None and "labels" in inputs:
   #         labels = inputs.pop("labels")
   #     else:
   #         labels = None
   #     outputs = model(**inputs)
   #     # Save past state if it exists
   #     # TODO: this needs to be fixed and made cleaner later.
   #     if self.args.past_index >= 0:
   #         self._past = outputs[self.args.past_index]


   #     if labels is not None:
   #         if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
   #             loss = self.label_smoother(outputs, labels, shift_labels=True)
   #         else:
   #             loss = self.label_smoother(outputs, labels)
   #     else:
   #         if isinstance(outputs, dict) and "loss" not in outputs:
   #             raise ValueError(
   #                 "The model did not return a loss from the inputs, only the following keys: "
   #                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
   #             )
   #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
   #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]


   #     return (loss, outputs) if return_outputs else loss
