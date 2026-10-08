# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke runs of the post-training entry points on tiny models, offline and on CPU.

Each run trains for two steps and checks the effect a user relies on: finite logged loss, and for
LoRA, a PEFT export on disk.
"""

import json
import math
from pathlib import Path

import jax
import jax.random as jrandom

import levanter.main.lora_lm as lora_lm
import levanter.main.train_dpo as train_dpo
from levanter.adaptor import LoraAdaptorConfig
from levanter.adaptor.lora import LoraConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, UrlDatasetSourceConfig
from levanter.data.text.preference import PreferenceChatLmDatasetFormat, PreferenceLmDataConfig
from levanter.distributed import DistributedConfig
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.testing import tiny_corpus
from levanter.testing.helpers import use_test_mesh
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import TrainerConfig

# Marks the assistant turn as the trained span, as the preference format requires.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{% endgeneration %}\n"
    "{% endif %}"
    "{% endfor %}"
)


def _trainer(tmp_path: Path) -> TrainerConfig:
    return TrainerConfig(
        num_train_steps=2,
        train_batch_size=len(jax.devices()),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        checkpointer=CheckpointerConfig(base_path=str(tmp_path / "checkpoints")),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
    )


def _logged_loss(tmp_path: Path, key: str) -> float:
    metrics = json.loads((tmp_path / "eval_results.json").read_text())
    assert key in metrics, f"no {key} among {sorted(metrics)}"
    return metrics[key]


def test_train_dpo_trains_a_lora_policy_against_its_base_model(tmp_path, local_gpt2_tokenizer_path):
    prefs = tmp_path / "prefs.jsonl"
    rows = [
        {
            "chosen": [{"role": "user", "content": f"Count to {i}."}, {"role": "assistant", "content": f"{i}"}],
            "rejected": [{"role": "user", "content": f"Count to {i}."}, {"role": "assistant", "content": "No."}],
        }
        for i in range(16)
    ]
    prefs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = PreferenceLmDataConfig(
        tokenizer=local_gpt2_tokenizer_path,
        components={
            "prefs": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[str(prefs)]),
                cache_dir=str(tmp_path / "cache"),
                format=PreferenceChatLmDatasetFormat(chat_template=CHAT_TEMPLATE),
            )
        },
        train_weights={"prefs": 1.0},
    )
    config = train_dpo.TrainDpoConfig(
        data=data,
        model=Gpt2Config(max_seq_len=32, hidden_dim=32, num_layers=2, num_heads=4),
        train_seq_len=32,
        trainer=_trainer(tmp_path),
        adapter=LoraAdaptorConfig(r=4),
        reference=train_dpo.AdapterBaseReferenceConfig(),
        validation_split_fraction=None,
    )

    train_dpo.main(config)

    assert math.isfinite(_logged_loss(tmp_path, "train/loss"))


def test_lora_lm_fine_tunes_a_hugging_face_checkpoint_and_exports_peft(tmp_path, local_gpt2_tokenizer_path):
    base = Gpt2Config(max_seq_len=64, hidden_dim=32, num_layers=2, num_heads=4, tokenizer=local_gpt2_tokenizer_path)
    converter = base.hf_checkpoint_converter()
    with use_test_mesh():
        model = Gpt2LMHeadModel.init(converter.Vocab, config=base, key=jrandom.PRNGKey(0))
        converter.save_pretrained(model, str(tmp_path / "base"), save_tokenizer=True)
    data, _ = tiny_corpus.construct_small_data_cache(str(tmp_path / "data"), tokenizer=local_gpt2_tokenizer_path)
    config = lora_lm.LoraLmConfig(
        initialize_from_hf=str(tmp_path / "base"),
        lora=LoraConfig(r=4, target_modules=["c_attn"]),
        data=data,
        trainer=_trainer(tmp_path),
        peft_save_path=str(tmp_path / "peft"),
        hf_save_steps=1,
        train_seq_len=64,
    )

    lora_lm.main(config)

    assert math.isfinite(_logged_loss(tmp_path, "train/loss"))
    assert list((tmp_path / "peft").rglob("adapter_config.json")), "no PEFT export was written"
