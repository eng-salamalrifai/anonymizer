# Importing each module runs its @register_model decorator.
# To add a model: create models/x.py, then add ONE import line here.
from models import bert_ner, nuner_zero  # noqa: F401
