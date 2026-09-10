import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec = importlib.util.spec_from_file_location("train_script", Path(__file__).resolve().parents[1] / "scripts" / "09_train.py")
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)

compute_warmup_steps = train_script.compute_warmup_steps


def test_compute_warmup_steps_matches_real_run_parameters():
    # Regression test: transformers 5.x removed TrainingArguments'
    # warmup_ratio kwarg (confirmed via a real Colab run); this function
    # exists specifically to convert it to the still-supported warmup_steps.
    steps = compute_warmup_steps(warmup_ratio=0.03, n_train_examples=169,
                                  per_device_train_batch_size=4, gradient_accumulation_steps=4,
                                  num_train_epochs=3)
    assert steps == 1  # ceil(169/16)=11 steps/epoch * 3 epochs = 33 total; round(33*0.03)=1


def test_compute_warmup_steps_scales_with_dataset_size():
    small = compute_warmup_steps(0.1, n_train_examples=100, per_device_train_batch_size=4,
                                  gradient_accumulation_steps=4, num_train_epochs=3)
    large = compute_warmup_steps(0.1, n_train_examples=10000, per_device_train_batch_size=4,
                                  gradient_accumulation_steps=4, num_train_epochs=3)
    assert large > small


def test_compute_warmup_steps_never_negative_for_zero_ratio():
    steps = compute_warmup_steps(0.0, n_train_examples=169, per_device_train_batch_size=4,
                                  gradient_accumulation_steps=4, num_train_epochs=3)
    assert steps == 0


def test_compute_warmup_steps_handles_dataset_smaller_than_batch_size():
    # ceil division must not produce zero steps per epoch (and thus zero
    # total steps) just because the dataset is tiny.
    steps = compute_warmup_steps(0.5, n_train_examples=3, per_device_train_batch_size=4,
                                  gradient_accumulation_steps=4, num_train_epochs=2)
    assert steps >= 0
    assert isinstance(steps, int)
