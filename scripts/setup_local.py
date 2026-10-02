import json,hashlib,os
from pathlib import Path
import yaml
from transformers import GemmaTokenizerFast
root=Path(__file__).resolve().parents[1]
source=Path(os.environ.get('PALIGEMMA_TOKENIZER_MODEL', '/root/.cache/openpi/big_vision/paligemma_tokenizer.model'))
output=root/'checkpoints/tokenizer'
tokenizer=GemmaTokenizerFast(vocab_file=str(source),add_bos_token=True,add_eos_token=False)
tokenizer.save_pretrained(output)
(output/'provenance.json').write_text(json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'note':'Reused official OpenPI PaliGemma SentencePiece vocabulary; GemmaTokenizerFast adds BOS and no EOS.'},indent=2))
base=root/'vendor/LIBERO/libero/libero'
config={'benchmark_root':str(base),'bddl_files':str(base/'bddl_files'),'init_states':str(base/'init_files'),
        'assets':str(base/'assets'),'datasets':str(root/'data/raw')}
p=root/'configs/libero'; p.mkdir(parents=True,exist_ok=True); (p/'config.yaml').write_text(yaml.safe_dump(config))
