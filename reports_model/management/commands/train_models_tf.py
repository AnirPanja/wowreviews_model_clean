# your_package/reviews/management/commands/train_models_tf.py
from django.core.management.base import BaseCommand
import subprocess
import os

class Command(BaseCommand):
    help = "Run TF preprocess + train token classifier + train relation model"

    def handle(self, *args, **options):
        print("Running preprocess...")
        subprocess.check_call(["python", "reports_model/preprocess_tf.py"])
        print("Training token classifier...")
        subprocess.check_call(["python", "reports_model/train_token_tf.py"])
        print("Training relation model...")
        subprocess.check_call(["python", "reports_model/train_relation_tf.py"])
        print("All done.")
