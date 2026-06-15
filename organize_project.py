from pathlib import Path
import shutil

root = Path(".")

folders = {
    "notebooks": [
        ".ipynb"
    ],

    "src/model": [
        "GPT_CONFIG_124M.py",
        "gpt_model.py",
        "transformer_block.py",
        "multi_head_attention.py",
        "feed_forward_network.py",
        "layer_normalization.py",
        "load_weights_into_gpt.py",
    ],

    "src/data": [
        "data_loader.py",
        "spam_dataset_prep.py",
        "token_converter.py",
        "text_to_id_to_text.py",
    ],

    "src/training": [
        "train_model_simple.py",
        "train_classifier_simple.py",
        "evaluate_model.py",
        "calc_loss_loader.py",
        "calc_loss_loader_classification.py",
        "calc_accuracy_loader.py",
        "gpt_download.py",
    ],

    "src/inference": [
        "generate.py",
        "generate_text_simple.py",
        "generate_and_print_sample.py",
    ],

    "src/utils": [
        "plot_losses.py",
        "plot_values.py",
        "bleu.py",
    ],

    "datasets": [
        "instruction-data.json",
        "instruction-data-with-response.json",
        "train.csv",
        "validation.csv",
        "test.csv",
        "sms_spam_collection.zip",
        "the-verdict.txt",
    ],

    "checkpoints": [
        ".pth"
    ],

    "outputs": [
        ".pdf"
    ]
}

for folder, patterns in folders.items():

    destination = root / folder
    destination.mkdir(parents=True, exist_ok=True)

    for pattern in patterns:

        # extension based
        if pattern.startswith("."):
            files = list(root.glob(f"*{pattern}"))

        else:
            files = [root / pattern]

        for file in files:

            if file.exists() and file.is_file():

                try:
                    shutil.move(str(file), str(destination / file.name))

                except shutil.Error:
                    pass

print("Organization complete.")