"""Offline compatibility checks using tiny RANDOM models, not perception benchmarks."""

import array
import importlib.util
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

from media_index.models import AudioEvents, SemanticModel, SpeechModel, VisionTags

AVAILABLE = all(
    importlib.util.find_spec(name)
    for name in ("torch", "transformers", "sentence_transformers")
)


@unittest.skipUnless(
    AVAILABLE, "Optional Torch/Transformers/SentenceTransformers extras not installed"
)
class LocalModelCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_audioset_sigmoid_and_emotion_softmax_local_loading(self):
        from transformers import (
            ASTConfig,
            ASTFeatureExtractor,
            ASTForAudioClassification,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wav = root / "tone.wav"
            with wave.open(str(wav), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(array.array("h", [1000, -1000] * 8000).tobytes())
            for task, labels in (
                ("audioset", ["Laughter", "Speech", "Applause"]),
                ("emotion", ["hap", "neu", "ang"]),
            ):
                folder = root / task
                config = ASTConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_hidden_layers=1,
                    num_mel_bins=16,
                    max_length=64,
                    patch_size=4,
                    frequency_stride=4,
                    time_stride=4,
                    num_labels=3,
                    id2label=dict(enumerate(labels)),
                )
                ASTForAudioClassification(config).save_pretrained(folder)
                ASTFeatureExtractor(num_mel_bins=16, max_length=64).save_pretrained(
                    folder
                )
                events = AudioEvents(folder, device="cpu", task=task).detect(
                    wav, threshold=0
                )
                self.assertEqual(len(events), 3)
                if task == "emotion":
                    self.assertAlmostEqual(
                        sum(event["confidence"] for event in events), 1, places=3
                    )
                    self.assertEqual(
                        {e["label"] for e in events}, {"joy", "neutral", "anger"}
                    )
                else:
                    self.assertGreater(
                        sum(event["confidence"] for event in events), 1.1
                    )
                    self.assertIn("laughter", {e["label"] for e in events})
                self.assertTrue(all(e["start"] == 0 and e["end"] == 1 for e in events))

    def test_faster_whisper_receives_pcm_instead_of_deprecated_pyav_decode(self):
        import numpy as np

        decoded = types.SimpleNamespace(transcribe=mock.Mock(return_value=([], None)))
        loader = mock.Mock(return_value=decoded)
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.dict(
                "sys.modules",
                {"faster_whisper": types.SimpleNamespace(WhisperModel=loader)},
            ),
        ):
            wav = Path(temp) / "tone.wav"
            with wave.open(str(wav), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(array.array("h", [1000, -1000] * 8000).tobytes())
            speech = SpeechModel("faster-whisper", temp)
            speech.transcribe(wav, 1)
            samples = decoded.transcribe.call_args.args[0]
            self.assertIsInstance(samples, np.ndarray)
            self.assertEqual(samples.dtype, np.float32)
            self.assertEqual(len(samples), 16000)
            self.assertTrue(decoded.transcribe.call_args.kwargs["vad_filter"])

    def test_clip_processor_and_local_visual_adapter(self):
        from tokenizers import ByteLevelBPETokenizer
        from transformers import (
            CLIPConfig,
            CLIPImageProcessorPil,
            CLIPModel,
            CLIPProcessor,
            CLIPTokenizer,
        )

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train_from_iterator(
                list(VisionTags.LABELS.values()),
                vocab_size=300,
                special_tokens=["<|startoftext|>", "<|endoftext|>"],
            )
            tokenizer.save_model(str(folder))
            clip_tokenizer = CLIPTokenizer(
                vocab=str(folder / "vocab.json"), merges=str(folder / "merges.txt")
            )
            config = CLIPConfig(
                projection_dim=16,
                text_config={
                    "vocab_size": len(clip_tokenizer),
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "max_position_embeddings": 77,
                    "bos_token_id": clip_tokenizer.bos_token_id,
                    "eos_token_id": clip_tokenizer.eos_token_id,
                },
                vision_config={
                    "image_size": 32,
                    "patch_size": 16,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                },
            )
            CLIPModel(config).save_pretrained(folder)
            image_processor = CLIPImageProcessorPil(
                size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
            )
            CLIPProcessor(
                tokenizer=clip_tokenizer, image_processor=image_processor
            ).save_pretrained(folder)
            tags = VisionTags(folder, "cpu").tag(bytes([128, 64, 32]) * 64 * 36, 64, 36)
            self.assertIsInstance(tags, list)
            self.assertTrue(all(0.4 <= tag["relative_similarity"] <= 1 for tag in tags))

    def test_sentence_transformer_query_document_apis(self):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer import modules as models
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bert = root / "bert"
            vocab = {
                word: i
                for i, word in enumerate(
                    ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hotel", "automation", "dog"]
                )
            }
            tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                pad_token="[PAD]",
                unk_token="[UNK]",
                cls_token="[CLS]",
                sep_token="[SEP]",
            ).save_pretrained(bert)
            BertModel(
                BertConfig(
                    vocab_size=len(vocab),
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                )
            ).save_pretrained(bert)
            transformer = models.Transformer(
                str(bert),
                model_kwargs={"local_files_only": True},
                processor_kwargs={"local_files_only": True},
            )
            sentence_model = SentenceTransformer(
                modules=[transformer, models.Pooling(16)]
            )
            folder = root / "sentence-model"
            sentence_model.save(str(folder))
            semantic = SemanticModel(folder, "cpu")
            documents = semantic.encode(["hotel automation", "dog"])
            queries = semantic.encode(["hotel automation"], query=True)
            self.assertEqual(len(documents), 2)
            self.assertEqual(len(queries[0]), 16)
            self.assertAlmostEqual(sum(v * v for v in queries[0]), 1, places=5)
