import os
import json
import random
import re
import difflib
from PIL import Image
from tqdm import tqdm
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
import pandas as pd
import random
import json
from PIL import Image
from io import BytesIO
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
import argparse
import fnmatch
import hashlib
import nltk
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

DEFAULT_TASK_DATA = (
    Path(__file__).resolve().parent
    / "data/MLLMU-Bench/Full_Set/train-00000-of-00001.parquet"
)
TASK_COLUMNS = ("Classification_Task", "Generation_Task", "Mask_Task")


def load_and_combine_parquet_files(directory):
    # Get all Parquet files in the directory
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in directory: {directory}")

    # Read and concatenate all Parquet files
    combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    return combined_df


def _read_eval_dataframe(path, task_data=None):
    """Read evaluation rows and enrich baseline metadata-only splits with tasks."""
    if os.path.isdir(path):
        df = load_and_combine_parquet_files(path)
    else:
        df = pd.read_parquet(path)
    missing = [column for column in TASK_COLUMNS if column not in df.columns]
    if not missing:
        return df
    task_path = Path(task_data or DEFAULT_TASK_DATA)
    if not task_path.is_file():
        raise FileNotFoundError(
            f"{path} has no task columns {missing}; task data not found at {task_path}"
        )
    task_df = pd.read_parquet(task_path)
    if "ID" not in task_df.columns or any(column not in task_df.columns for column in missing):
        raise ValueError(f"Task data {task_path} must contain ID and {missing}")
    lookup = task_df[["ID", *missing]].drop_duplicates("ID")
    enriched = df.drop(columns=[column for column in missing if column in df.columns], errors="ignore").merge(
        lookup, on="ID", how="left", validate="many_to_one"
    )
    if enriched[missing].isna().any().any():
        missing_ids = enriched.loc[enriched[missing].isna().any(axis=1), "ID"].tolist()[:5]
        raise ValueError(f"Task data {task_path} has no rows for IDs such as {missing_ids}")
    return enriched

def save_ids_to_json(parquet_file, output_folder, filename="ids.json"):
    """
    Extract IDs from a Parquet file and save them to a JSON file in the specified folder.

    Args:
        parquet_file (str): Path to the Parquet file containing the data.
        output_folder (str): Path to the folder where the JSON file will be saved.
        filename (str): Name of the JSON file. Defaults to "ids.json".
    """
    # Load the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)

    # Extract the unique IDs
    ids = df['ID'].unique().tolist()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full path to the JSON file
    output_json_file = os.path.join(output_folder, filename)

    # Save the IDs to a JSON file
    with open(output_json_file, 'w') as f:
        json.dump(ids, f)

    print(f"Saved IDs to {output_json_file}")

def compute_bleu(ground_truth, predicted_answer):
    """
    Compute the BLEU score between a ground truth and predicted answer using simple whitespace tokenization.

    Args:
        ground_truth (str): The correct reference answer.
        predicted_answer (str): The predicted answer from the model.

    Returns:
        float: The BLEU score.
    """
    # Use .split() to tokenize based on spaces
    reference = [ground_truth.split()]  # Reference needs to be a list of tokenized words
    hypothesis = predicted_answer.split()  # Hypothesis (predicted answer) is also tokenized

    # Use smoothing to handle cases where BLEU score could be 0 for short texts
    smoothing_function = SmoothingFunction().method1

    # Compute the BLEU score
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

    return bleu_score

def evaluate_from_ids(id_json_file, question_folder, filename_pattern="*"):
    """
    Load IDs from the JSON file and find their corresponding evaluation question files with a specific filename pattern,
    then return a list of the loaded JSON files.

    Args:
        id_json_file (str): Path to the JSON file containing the list of IDs.
        question_folder (str): Path to the folder containing evaluation question files.
        filename_pattern (str): Filename pattern to match (e.g., "*_question.json"). Default is "*" for any file.

    Returns:
        list: A list of loaded JSON files from the question folder.
    """
    # Load the list of IDs from the ID JSON file
    with open(id_json_file, 'r') as f:
        ids = json.load(f)

    json_files = []

    # Loop through the files in the question folder
    for filename in sorted(os.listdir(question_folder)):
        # Find files that match the ID and the filename pattern
        for id_ in ids:
            if filename.startswith(id_) and fnmatch.fnmatch(filename, filename_pattern):
                file_path = os.path.join(question_folder, filename)

                # Load the matching JSON file
                with open(file_path, 'r') as f:
                    json_files.append(json.load(f))
                break  # Move to the next file after finding the match

    return json_files

def formulate_prompt_with_options(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"({key}) {value}" for key, value in options.items()])
    prompt = f"Question: {question}\nOptions:\n{options_str}"
    return prompt


def build_zero_shot_prompt(question, include_image):
    """Format a zero-shot prompt to match the LLaVA training conversation."""
    user_prefix = "USER: <image>\n" if include_image else "USER:\n"
    return f"{user_prefix}{question}\nASSISTANT:"


def build_model_prompt(question, include_image, processor, model_id):
    """Build a model-native zero-shot conversation prompt."""
    if is_qwen_model(model_id, processor):
        content = []
        if include_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return build_zero_shot_prompt(question, include_image)


def is_qwen_model(model_id, processor=None):
    """Identify Qwen2.5-VL from a model ID/path or loaded processor."""
    model_name = str(model_id).casefold()
    if "qwen2.5-vl" in model_name or "qwen_vanilla" in model_name:
        return True
    return processor is not None and processor.__class__.__name__.startswith("Qwen2_5_VL")


def build_qwen_few_shot_prompt(processor, demonstrations, question, image=None):
    """Render optional demonstrations and the query with Qwen's chat template."""
    messages = []
    images = []
    for demo_question, demo_answer, demo_image in demonstrations:
        content = []
        if demo_image is not None:
            content.append({"type": "image"})
            images.append(demo_image)
        content.append({"type": "text", "text": demo_question})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": demo_answer})
    content = []
    if image is not None:
        content.append({"type": "image"})
        images.append(image)
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, images


def decode_generated_answer(inputs, outputs, decoder):
    """Decode only tokens generated after the input prompt."""
    input_length = inputs["input_ids"].shape[-1]
    answer = decoder.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
    for marker in ("ASSISTANT:", "Answer:"):
        if marker in answer:
            answer = answer.rsplit(marker, 1)[-1].strip()
    return answer


def extract_choice(answer, options):
    """Extract an option key from concise or mildly verbose model output."""
    option_keys = {str(key).upper() for key in options}

    # Prefer an explicit answer marker so letters in explanatory text do not win.
    marked = re.search(
        r"(?:ANSWER|CHOICE|OPTION)\s*(?:IS|:)??\s*[\(\[]?([A-Z])[\)\].,:]?",
        answer.upper(),
    )
    if marked and marked.group(1) in option_keys:
        return marked.group(1)

    # Most compliant generations are just "A", "(A)", or "A.".
    first = re.match(r"^\s*[\(\[]?([A-Z])[\)\].,:]?\s*$", answer.upper())
    if first and first.group(1) in option_keys:
        return first.group(1)

    for match in re.finditer(r"(?<![A-Z0-9])([A-Z])(?![A-Z0-9])", answer.upper()):
        if match.group(1) in option_keys:
            return match.group(1)
    return None


def select_answer(assistant_response, option_values):
    """Soft-match a generated response to the closest option text."""
    if not option_values:
        raise ValueError("Classification question has no options")

    response = " ".join(str(assistant_response).split()).casefold()
    normalized_options = [" ".join(str(value).split()) for value in option_values]
    similarities = [
        difflib.SequenceMatcher(None, response, value.casefold()).ratio()
        for value in normalized_options
    ]
    return normalized_options[similarities.index(max(similarities))]


def answer_contains_correct_option(assistant_response, correct_answer):
    """Return whether the correct option text occurs in the raw response."""
    response = " ".join(str(assistant_response).split()).casefold()
    answer = " ".join(str(correct_answer).split()).casefold()
    if not answer:
        return False

    prefix = r"(?<!\w)" if answer[0].isalnum() else ""
    suffix = r"(?!\w)" if answer[-1].isalnum() else ""
    return re.search(f"{prefix}{re.escape(answer)}{suffix}", response) is not None


class InvalidClassificationAnswer(ValueError):
    """Raised when a benchmark ground truth is absent from its option set."""


def classification_answers(options, correct_answer):
    """Return ordered option texts and the correct option text."""
    option_values = [str(value) for value in options.values()]
    correct_key = str(correct_answer).split(".", 1)[0].strip().upper()
    option_by_key = {str(key).upper(): str(value) for key, value in options.items()}
    if correct_key in option_by_key:
        return option_values, option_by_key[correct_key]

    normalized_answer = " ".join(str(correct_answer).split()).casefold()
    for option_value in option_values:
        if " ".join(option_value.split()).casefold() == normalized_answer:
            return option_values, option_value
    raise InvalidClassificationAnswer(
        f"Ground truth {correct_answer!r} is absent from options={options}"
    )


def _model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def generate_answer(question, include_image, image, processor, tokenizer, model, args, max_new_tokens):
    """Generate one answer using the correct prompt and device for the selected model."""
    prompt = build_model_prompt(question, include_image, processor, args.model_id)
    if is_qwen_model(args.model_id, processor):
        processor_kwargs = {"text": prompt, "return_tensors": "pt"}
        if include_image:
            processor_kwargs["images"] = [image]
        inputs = processor(**processor_kwargs)
        decoder = tokenizer
    elif include_image:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        decoder = tokenizer
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        decoder = tokenizer
    inputs = inputs.to(_model_input_device(model))
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return prompt, decode_generated_answer(inputs, outputs, decoder)


def _limit_eval_samples(eval_samples, args):
    if args.max_eval_samples is None:
        return eval_samples
    return eval_samples.head(args.max_eval_samples)


def _limit_questions(questions, args):
    if args.max_questions_per_group is None:
        return questions
    return questions[:args.max_questions_per_group]


def select_row_image_bytes(row, *, mode, excluded_image_sha256_by_id=None):
    """Select a test pose while excluding any pose disclosed during training."""
    if mode != "test" or "images" not in row:
        return row["image"]["bytes"]
    candidates = list(row["images"])
    if excluded_image_sha256_by_id:
        entity_id = str(row["ID"]).zfill(3)
        excluded = set(excluded_image_sha256_by_id.get(entity_id, ()))
        if excluded:
            candidates = [
                value
                for value in candidates
                if hashlib.sha256(value["bytes"]).hexdigest() not in excluded
            ]
    if not candidates:
        raise ValueError(f"No test pose remains for ID {row['ID']} after exclusions")
    return random.choice(candidates)["bytes"]


def formulate_prompt_with_options_llama(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n####Choices:\n{options_str}"
    return prompt
def split_dataset(original_dataset, forget_percentage=0.3):
    forget_set_size = int(len(original_dataset) * forget_percentage)
    retain_set_size = len(original_dataset) - forget_set_size
    forget_set, retain_set = train_test_split(original_dataset, test_size=retain_set_size, random_state=42)
    return forget_set, retain_set

def load_json_files(question_folder):
    """
    Load all JSON files from the given folder.
    """
    json_files = []
    for filename in sorted(os.listdir(question_folder)):
        if filename.endswith(".json"):
            with open(os.path.join(question_folder, filename), 'r') as f:
                json_files.append(json.load(f))
    return json_files

def load_image(image_folder, image_id):
    """
    Load an image, trying both .png and .jpg extensions.
    """
    possible_extensions = ['.png', '.jpg', '.jpeg']
    for ext in possible_extensions:
        image_path = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                return image
            except Exception as e:
                print(f"Error loading image at {image_path}: {e}")
                return None
    print(f"Image not found for ID: {image_id}")
    return None


def load_random_test_image(image_folder, image_id):
    """
    Load a random image from a folder in 'test' mode.

    Args:
        image_folder: The folder where the image_id folder is stored.
        image_id: The ID of the folder containing multiple images.

    Returns:
        image: The randomly selected image (or None if not found or error occurs).
    """
    # In 'test' mode, image_id is a folder containing multiple images
    image_dir = os.path.join(image_folder, image_id)

    if not os.path.isdir(image_dir):
        print(f"Image folder not found for ID: {image_id}")
        return None

    # List the images inside the folder
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Filter to match the expected naming format: {image_id}_poseX_gen1.png
    image_files = [f for f in image_files if f.startswith(image_id) and 'pose' in f]

    if not image_files:
        print(f"No valid images found in folder: {image_dir}")
        return None

    # Randomly select one image from the folder
    selected_image = random.choice(image_files)
    image_path = os.path.join(image_dir, selected_image)

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Randomly selected image: {selected_image}")
        return image
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return None

def evaluate_classification(
    parquet_file,
    processor,
    tokenizer,
    model,
    args,
    id_list_file=None,
    mode="default",
    forget_parquet_file=None,
    few_shot_parquet_file=None,
    question_type_filter=None,
    excluded_image_sha256_by_id=None,
):
    """
    Evaluate classification task with the benchmark's one-person/two-person
    few-shot prompt when ``few_shot_parquet_file`` is supplied.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        processor: The processor for handling image and text inputs.
        tokenizer: The tokenizer for decoding model outputs.
        model: The model to use for classification.
        args: Arguments object containing model ID and other configurations.
        id_list_file: (Optional) Path to the JSON file containing the list of IDs. Default is None.
        mode: Evaluation subset mode ('forget', 'retain_share', 'test', or others).
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary with accuracy scores.
    """
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {mode} Mode #########################################" )

    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the main Parquet file
        df = _read_eval_dataframe(parquet_file, args.task_data)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0
    skipped_image_textual_questions = 0
    skipped_pure_text_questions = 0

    # Reproduce the original benchmark few-shot protocol. For LLaVA, two
    # people are demonstrations; their questions are excluded from scoring.
    few_shot_images = []
    few_shot_image_prompts = []
    few_shot_text_prompts = []
    few_shot_question_indices = {}
    selected_ids = []
    if few_shot_parquet_file:
        available_ids = list(id_list)
        # The original MLLMU-Bench protocol uses exactly one demonstration person.
        n_shot = 1
        n_shot = min(n_shot, len(available_ids))
        selected_ids = random.sample(available_ids, n_shot)
        few_shot_df = pd.read_parquet(few_shot_parquet_file)
        few_shot_samples = few_shot_df[few_shot_df["ID"].isin(selected_ids)]
        for _, shot_row in few_shot_samples.iterrows():
            shot_id = shot_row["ID"]
            shot_questions = shot_row["Classification_Task"]
            shot_image = Image.open(BytesIO(shot_row["image"]["bytes"])).convert("RGB")
            few_shot_question_indices[shot_id] = {"image_textual": [], "pure_text": []}
            for idx, item in enumerate(shot_questions.get("Image_Textual_Questions", [])):
                few_shot_image_prompts.append(item)
                few_shot_images.append(shot_image)
                few_shot_question_indices[shot_id]["image_textual"].append(idx)
            for idx, item in enumerate(shot_questions.get("Pure_Text_Questions", [])):
                few_shot_text_prompts.append(item)
                few_shot_question_indices[shot_id]["pure_text"].append(idx)
        print(f"Selected few-shot IDs: {selected_ids}")
        print(f"Loaded {len(few_shot_image_prompts)} image and {len(few_shot_text_prompts)} text demonstrations.")

    def shot_prompt(item):
        options = item["Options"]
        option_values, correct_answer = classification_answers(options, item["Correct_Answer"])
        return (
            f"{item['Question']}\nSelect answer in {option_values}\n"
            f"Correct Answer: {correct_answer}\n"
        )

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = _read_eval_dataframe(parquet_file, args.task_data)
        else:
            df = _read_eval_dataframe(parquet_file, args.task_data)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = _read_eval_dataframe(parquet_file, args.task_data)
        eval_samples = df[df['ID'].isin(id_list)]
    eval_samples = _limit_eval_samples(eval_samples, args)

    # Process each evaluation sample
    for _, row in eval_samples.iterrows():
        classification_questions = row["Classification_Task"]

        # Randomly select one image if in test mode
        image_data = select_row_image_bytes(
            row,
            mode=mode,
            excluded_image_sha256_by_id=excluded_image_sha256_by_id,
        )

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Iterate through each image-textual question
        print("########################## Processing Image-Textual Questions ########################## ")
        for idx, question_data in enumerate(_limit_questions(classification_questions.get("Image_Textual_Questions", []), args)):
            if question_type_filter not in (None, "Image_Textual"):
                continue
            if row["ID"] in few_shot_question_indices and idx in few_shot_question_indices[row["ID"]]["image_textual"]:
                continue
            question = question_data["Question"]
            options = question_data["Options"]
            try:
                option_values, correct_answer = classification_answers(
                    options, question_data["Correct_Answer"]
                )
            except InvalidClassificationAnswer as exc:
                skipped_image_textual_questions += 1
                print(f"Skipping invalid classification question for ID {row['ID']}: {exc}")
                continue
            question_with_options = f"{question}\nSelect answer in {option_values}"

            if few_shot_parquet_file:
                if is_qwen_model(args.model_id, processor):
                    prompt, prompt_images = build_qwen_few_shot_prompt(
                        processor,
                        [(shot_prompt(item), shot_prompt(item).split("Correct Answer: ", 1)[-1].strip(), shot_image)
                         for item, shot_image in zip(few_shot_image_prompts, few_shot_images)],
                        question_with_options,
                        image,
                    )
                    inputs = processor(
                        images=prompt_images, text=prompt, return_tensors="pt"
                    ).to(_model_input_device(model))
                else:
                    prompt = "".join(
                        f"USER: <image>\n{shot_prompt(item)}"
                        for item in few_shot_image_prompts
                    )
                    prompt += (
                        f"USER: <image>\n{question_with_options}\n"
                        "ASSISTANT:"
                    )
                    inputs = processor(
                        images=[*few_shot_images, image], text=prompt, return_tensors="pt"
                    ).to(_model_input_device(model))
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                assistant_response = decode_generated_answer(inputs, outputs, tokenizer)
            else:
                prompt, assistant_response = generate_answer(
                    question_with_options,
                    True, image, processor, tokenizer, model, args, max_new_tokens=50,
                )
            answer_is_correct = answer_contains_correct_option(
                assistant_response, correct_answer
            )
            if answer_is_correct:
                total_image_textual_correct += 1
            total_image_textual_questions += 1
            print("Prompt: ", prompt)
            print("Raw Model Answer: ", assistant_response)
            print("Correct Answer: ", correct_answer)
            print("Correct Answer Found in Raw Response: ", answer_is_correct)
            print("\n")

        # Process Pure_Text_Questions
        print("########################## Processing Pure-textual Questions ########################## ")
        for idx, question_data in enumerate(_limit_questions(classification_questions.get("Pure_Text_Questions", []), args)):
            if question_type_filter not in (None, "Pure_Text"):
                continue
            if row["ID"] in few_shot_question_indices and idx in few_shot_question_indices[row["ID"]]["pure_text"]:
                continue
            question = question_data["Question"]
            options = question_data["Options"]
            try:
                option_values, correct_answer = classification_answers(
                    options, question_data["Correct_Answer"]
                )
            except InvalidClassificationAnswer as exc:
                skipped_pure_text_questions += 1
                print(f"Skipping invalid classification question for ID {row['ID']}: {exc}")
                continue
            question_with_options = f"{question}\nSelect answer in {option_values}"

            if few_shot_parquet_file:
                if is_qwen_model(args.model_id, processor):
                    prompt, _ = build_qwen_few_shot_prompt(
                        processor,
                        [(shot_prompt(item), shot_prompt(item).split("Correct Answer: ", 1)[-1].strip(), None)
                         for item in few_shot_text_prompts],
                        question_with_options,
                    )
                    inputs = processor(text=prompt, return_tensors="pt").to(_model_input_device(model))
                else:
                    prompt = "".join(
                        f"USER:\n{shot_prompt(item)}"
                        for item in few_shot_text_prompts
                    )
                    prompt += (
                        f"USER:\n{question_with_options}\n"
                        "ASSISTANT:"
                    )
                    inputs = tokenizer(prompt, return_tensors="pt").to(_model_input_device(model))
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                assistant_response = decode_generated_answer(inputs, outputs, tokenizer)
            else:
                prompt, assistant_response = generate_answer(
                    question_with_options,
                    False, None, processor, tokenizer, model, args, max_new_tokens=50,
                )
            answer_is_correct = answer_contains_correct_option(
                assistant_response, correct_answer
            )
            if answer_is_correct:
                total_pure_text_correct += 1
            total_pure_text_questions += 1

            print("Prompt: ", prompt)
            print("Raw Model Answer: ", assistant_response)
            print("Correct Answer: ", correct_answer)
            print("Correct Answer Found in Raw Response: ", answer_is_correct)
            print("\n")

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")
    print(f"Skipped Invalid Image-Textual Questions: {skipped_image_textual_questions}")
    print(f"Skipped Invalid Pure Text Questions: {skipped_pure_text_questions}")

    return {
        "Image-Textual Question Accuracy": image_textual_accuracy,
        "Pure Text Question Accuracy": pure_text_accuracy,
        "Skipped Invalid Image-Textual Questions": skipped_image_textual_questions,
        "Skipped Invalid Pure Text Questions": skipped_pure_text_questions,
    }


# def evaluate_fill_in_the_blank(json_files, image_folder, processor, tokenizer, model, args, id_list_file=None, mode="default"):
def evaluate_fill_in_the_blank(
    parquet_file,
    processor,
    tokenizer,
    model,
    args,
    id_list_file=None,
    mode="default",
    forget_parquet_file=None,
    few_shot_parquet_file=None,
    question_type_filter=None,
    excluded_image_sha256_by_id=None,
):
    """
    Evaluate fill-in-the-blank tasks with zero-shot prompts.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        processor: The processor for handling image and text inputs.
        tokenizer: The tokenizer for decoding model outputs.
        model: The model to use for classification.
        args: Arguments object containing model ID and other configurations.
        id_list_file: (Optional) Path to the JSON file containing the list of IDs. Default is None.
        mode: Evaluation subset mode ('forget', 'retain_share', 'test', or others).
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary with accuracy scores.
    """
    print(
        "################################## Fill-in-the-blank Task Starts ##############################################")

    print(f"Evaluating {mode} Mode")
    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the Parquet file
        df = _read_eval_dataframe(parquet_file, args.task_data)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    few_shot_images = []
    few_shot_image_prompts = []
    few_shot_text_prompts = []
    few_shot_question_indices = {}
    if few_shot_parquet_file:
        # Use one demonstration person for the explicit one-shot protocol.
        n_shot = 1
        selected_ids = random.sample(list(id_list), min(n_shot, len(id_list)))
        few_shot_df = pd.read_parquet(few_shot_parquet_file)
        for _, shot_row in few_shot_df[few_shot_df["ID"].isin(selected_ids)].iterrows():
            shot_id = shot_row["ID"]
            shot_image = Image.open(BytesIO(shot_row["image"]["bytes"])).convert("RGB")
            few_shot_question_indices[shot_id] = {"image_textual": [], "pure_text": []}
            for idx, item in enumerate(shot_row["Mask_Task"]):
                example = {
                    "Question": item["Question"].replace("__", "[Blank]")
                    + "\nPlease **ONLY** provide the correct answer that should replace the [Blank].",
                    "Correct Answer": item["Ground_Truth"],
                }
                if item["Type"] == "Image_Textual":
                    few_shot_image_prompts.append(example)
                    few_shot_images.append(shot_image)
                    few_shot_question_indices[shot_id]["image_textual"].append(idx)
                else:
                    few_shot_text_prompts.append(example)
                    few_shot_question_indices[shot_id]["pure_text"].append(idx)
        print(f"Selected few-shot IDs: {selected_ids}")
        print(f"Loaded {len(few_shot_image_prompts)} image and {len(few_shot_text_prompts)} text cloze demonstrations.")

    # Load evaluation samples
    # Load the test set with multiple Parquet files if mode is "test"
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = _read_eval_dataframe(parquet_file, args.task_data)
        else:
            df = _read_eval_dataframe(parquet_file, args.task_data)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = _read_eval_dataframe(parquet_file, args.task_data)
        eval_samples = df[df['ID'].isin(id_list)]
    eval_samples = _limit_eval_samples(eval_samples, args)

    # Process each evaluation sample
    for _, row in eval_samples.iterrows():
        fill_in_the_blank_questions = row["Mask_Task"]

        # Randomly select one image if in test mode
        image_data = select_row_image_bytes(
            row,
            mode=mode,
            excluded_image_sha256_by_id=excluded_image_sha256_by_id,
        )

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Iterate through each question in Mask_Task.
        for idx, question_entry in enumerate(_limit_questions(fill_in_the_blank_questions, args)):
            if question_type_filter is not None and question_entry["Type"] != question_type_filter:
                continue
            if row["ID"] in few_shot_question_indices:
                group = "image_textual" if question_entry["Type"] == "Image_Textual" else "pure_text"
                if idx in few_shot_question_indices[row["ID"]][group]:
                    continue
            question = question_entry["Question"]
            ground_truth = question_entry["Ground_Truth"]
            question_type = question_entry["Type"]
            question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."

            if few_shot_parquet_file:
                if question_type == "Image_Textual":
                    if is_qwen_model(args.model_id, processor):
                        prompt, prompt_images = build_qwen_few_shot_prompt(
                            processor,
                            [(item["Question"], item["Correct Answer"], shot_image)
                             for item, shot_image in zip(few_shot_image_prompts, few_shot_images)],
                            question,
                            image,
                        )
                        inputs = processor(
                            images=prompt_images, text=prompt, return_tensors="pt"
                        ).to(_model_input_device(model))
                    else:
                        examples = "".join(
                            f"USER: <image>\n{item['Question']}\n"
                            f"ASSISTANT:{item['Correct Answer']}\n"
                            for item in few_shot_image_prompts
                        )
                        prompt = (
                            f"{examples}USER: <image>\n{question}\nASSISTANT:"
                        )
                        inputs = processor(
                            images=[*few_shot_images, image], text=prompt, return_tensors="pt"
                        ).to(_model_input_device(model))
                    decoder = tokenizer
                else:
                    if is_qwen_model(args.model_id, processor):
                        prompt, _ = build_qwen_few_shot_prompt(
                            processor,
                            [(item["Question"], item["Correct Answer"], None)
                             for item in few_shot_text_prompts],
                            question,
                        )
                        inputs = processor(text=prompt, return_tensors="pt").to(_model_input_device(model))
                    else:
                        examples = "".join(
                            f"USER:\n{item['Question']}\n"
                            f"ASSISTANT:{item['Correct Answer']}\n"
                            for item in few_shot_text_prompts
                        )
                        prompt = f"{examples}USER:\n{question}\nASSISTANT:"
                        inputs = tokenizer(prompt, return_tensors="pt").to(_model_input_device(model))
                    decoder = tokenizer
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                assistant_response = decode_generated_answer(inputs, outputs, decoder)
            else:
                prompt, assistant_response = generate_answer(
                    question,
                    question_type == "Image_Textual",
                    image if question_type == "Image_Textual" else None,
                    processor,
                    tokenizer,
                    model,
                    args,
                    max_new_tokens=50,
                )

            print("Prompt: ", prompt)
            print("Model Answer: ", assistant_response)
            print("Correct Answer: ", ground_truth)
            print("The model answer is: ", ground_truth.lower() in assistant_response.lower())
            print("\n")
            # Evaluate if the generated answer contains the correct ground truth
            if question_type == "Image_Textual":
                if ground_truth.lower() in assistant_response.lower():
                    total_image_textual_correct += 1
                total_image_textual_questions += 1
            elif question_type == "Pure_Text":
                if ground_truth.lower() in assistant_response.lower():
                    total_pure_text_correct += 1
                total_pure_text_questions += 1

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")

    return {
        "image_textual_accuracy": image_textual_accuracy,
        "pure_text_accuracy": pure_text_accuracy
    }

def evaluate_generation(
    parquet_file,
    processor,
    tokenizer,
    model,
    args,
    mode="default",
    forget_parquet_file=None,
    question_type_filter=None,
    excluded_image_sha256_by_id=None,
):
    """
    Evaluate the generation tasks using the ROUGE and BLEU scores.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        processor: The processor for handling text and images (e.g., from Hugging Face).
        tokenizer: The tokenizer for decoding model outputs.
        model: The model for answering the generation questions.
        args: Arguments object containing model ID and other configurations.
        file_name: Name of the file to save the evaluation results.
        mode: Mode to control which evaluation setup to use. Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary containing average ROUGE and BLEU scores for Image_Textual and Pure_Text questions.
    """
    print("################################## Generation Task Starts ##############################################")

    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize variables to store scores and question counts for both question types
    total_rouge1_img = total_rouge2_img = total_rougeL_img = total_bleu_img = total_image_textual_questions = 0
    total_rouge1_text = total_rouge2_text = total_rougeL_text = total_bleu_text = total_pure_text_questions = 0

    # Initialize list to store the results
    results = {
        "Generation_Questions": []
    }

    # Load the ID list from the forget Parquet file for filtering if mode is "test"
    if mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # Load all IDs from the Parquet file if no filtering is needed
        df = _read_eval_dataframe(parquet_file, args.task_data)
        id_list = df['ID'].unique().tolist()

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = _read_eval_dataframe(parquet_file, args.task_data)
        else:
            df = _read_eval_dataframe(parquet_file, args.task_data)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = _read_eval_dataframe(parquet_file, args.task_data)
        eval_samples = df[df['ID'].isin(id_list)]
    eval_samples = _limit_eval_samples(eval_samples, args)

    # Loop through each person's data in the evaluation samples
    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        image_id = row["ID"]
        generation_questions = row["Generation_Task"]

        # Randomly select one image if in test mode and multiple images are available
        image_data = select_row_image_bytes(
            row,
            mode=mode,
            excluded_image_sha256_by_id=excluded_image_sha256_by_id,
        )

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Process each generation question
        for question_data in _limit_questions(generation_questions, args):
            question_type = question_data["Type"]
            if question_type_filter is not None and question_type != question_type_filter:
                continue
            question = question_data["Question"]
            ground_truth = question_data["Ground_Truth"]

            if question_type == "Image_Textual":
                prompt, predicted_answer = generate_answer(
                    f"{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH.",
                    True,
                    image,
                    processor,
                    tokenizer,
                    model,
                    args,
                    max_new_tokens=50,
                )

            else:  # Pure_Text case
                prompt, predicted_answer = generate_answer(
                    f"{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH.",
                    False,
                    None,
                    processor,
                    tokenizer,
                    model,
                    args,
                    max_new_tokens=50,
                )

            # Print debug information
            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "image_id": image_id,
                "question type": question_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE and BLEU scores
            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            if question_type == "Image_Textual":
                # Accumulate scores for Image_Textual questions
                total_bleu_img += bleu_score
                total_rouge1_img += rouge_scores['rouge1'].fmeasure
                total_rouge2_img += rouge_scores['rouge2'].fmeasure
                total_rougeL_img += rouge_scores['rougeL'].fmeasure
                total_image_textual_questions += 1
            else:
                # Accumulate scores for Pure_Text questions
                total_bleu_text += bleu_score
                total_rouge1_text += rouge_scores['rouge1'].fmeasure
                total_rouge2_text += rouge_scores['rouge2'].fmeasure
                total_rougeL_text += rouge_scores['rougeL'].fmeasure
                total_pure_text_questions += 1

    # Save the results to a JSON file
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(f'{args.output_folder}/{mode}_generation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Calculate and print average ROUGE and BLEU scores
    avg_scores = {}
    if total_image_textual_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Image_Textual)": total_rouge1_img / total_image_textual_questions,
            "Average ROUGE-2 (Image_Textual)": total_rouge2_img / total_image_textual_questions,
            "Average ROUGE-L (Image_Textual)": total_rougeL_img / total_image_textual_questions,
            "Average BLEU (Image_Textual)": total_bleu_img / total_image_textual_questions
        })

    if total_pure_text_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Pure_Text)": total_rouge1_text / total_pure_text_questions,
            "Average ROUGE-2 (Pure_Text)": total_rouge2_text / total_pure_text_questions,
            "Average ROUGE-L (Pure_Text)": total_rougeL_text / total_pure_text_questions,
            "Average BLEU (Pure_Text)": total_bleu_text / total_pure_text_questions
        })

    for metric, score in avg_scores.items():
        print(f"{metric}: {score}")

    return avg_scores


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to the model.')
    parser.add_argument('--cache_path', type=str, help='Local trained-model checkpoint path.')
    parser.add_argument(
        '--processor_path',
        type=str,
        default=None,
        help='Optional local processor path, useful when model weights omit tokenizer files.',
    )
    parser.add_argument('--data_split_folder', type=str, required=True, help='Forget/retain split root directory.')
    parser.add_argument(
        '--task_data',
        type=str,
        default=str(DEFAULT_TASK_DATA),
        help='Full-set parquet used to enrich metadata-only baseline splits.',
    )
    parser.add_argument('--few_shot_data', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--test_data', type=str, required=True, help='Test Parquet file or directory.')
    parser.add_argument('--celebrity_data', type=str, required=True, help='Real-celebrity Parquet file.')
    parser.add_argument('--output_folder', type=str, required=True, help='Evaluation output directory.')
    parser.add_argument('--output_file', type=str, required=True, help='Output filename prefix.')
    parser.add_argument('--forget_ratio', type=int, default=5, help='Forget split percentage.')
    parser.add_argument('--pretrain', action='store_true', help="Evaluate the model from --model_id instead of --cache_path.")
    parser.add_argument(
        '--attn_implementation',
        choices=('flash_attention_2', 'sdpa', 'eager'),
        default='flash_attention_2',
        help='Attention backend for Qwen2.5-VL.',
    )
    parser.add_argument('--max_eval_samples', type=int, default=None, help='Optional per-dataset row limit for smoke tests.')
    parser.add_argument('--max_questions_per_group', type=int, default=None, help='Optional per-task question limit for smoke tests.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for test-image selection.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA-capable GPU.")
    if not (1 <= args.forget_ratio <= 99):
        raise ValueError("--forget_ratio must be between 1 and 99.")
    if args.max_eval_samples is not None and args.max_eval_samples < 1:
        raise ValueError("--max_eval_samples must be at least 1 when provided.")
    if args.max_questions_per_group is not None and args.max_questions_per_group < 1:
        raise ValueError("--max_questions_per_group must be at least 1 when provided.")
    if not args.pretrain and not args.cache_path:
        raise ValueError("--cache_path is required unless --pretrain is used.")
    qwen_model = is_qwen_model(args.model_id)
    llava_model = args.model_id.casefold().startswith("llava")
    if not (qwen_model or llava_model):
        raise ValueError("Unsupported --model_id. Expected a LLaVA or Qwen2.5-VL model ID.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Construct folder paths for "forget" and "retain"
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{100 - args.forget_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)
    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")
    required_paths = [forget_parquet_file, retain_parquet_file, args.test_data, args.celebrity_data, args.task_data]
    if not args.pretrain:
        required_paths.append(args.cache_path)
    for required_path in required_paths:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required evaluation path not found: {required_path}")
    os.makedirs(args.output_folder, exist_ok=True)

    if args.processor_path:
        processor_source = args.processor_path
    elif (
        qwen_model
        and ("qwen_vanilla" in args.model_id.casefold() or os.path.isdir(args.model_id))
        and (Path(__file__).resolve().parent / "Qwen_Vanilla/processor").is_dir()
    ):
        processor_source = str(Path(__file__).resolve().parent / "Qwen_Vanilla/processor")
    else:
        processor_source = args.model_id
    processor_is_local = os.path.isdir(processor_source)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        local_files_only=processor_is_local,
    )
    tokenizer = processor.tokenizer
    if qwen_model:
        # Qwen2.5-VL with FlashAttention 2 requires left padding for batches.
        tokenizer.padding_side = "left"

    torch.cuda.empty_cache()
    model_source = args.model_id if args.pretrain else args.cache_path
    model_is_local = os.path.isdir(model_source)
    model_config = AutoConfig.from_pretrained(model_source, local_files_only=model_is_local)
    if getattr(model_config, "model_type", "") == "qwen2_5_vl":
        print("Loading Qwen2.5-VL model with FlashAttention 2...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=model_is_local,
            attn_implementation=args.attn_implementation,
        )
    elif getattr(model_config, "model_type", "").startswith("llava"):
        print("Loading LLaVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=model_is_local,
        )
    else:
        raise ValueError(f"Unsupported model_type={getattr(model_config, 'model_type', None)!r}")
    model.eval()


    # Evaluate Forget Set (from shared classification and generation folders)
    torch.cuda.empty_cache()
    print("### Evaluating Forget Set ###")
    forget_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=forget_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget")

    forget_classification_result = evaluate_classification(parquet_file=forget_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget")

    forget_generation_result = evaluate_generation(parquet_file=forget_parquet_file,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="forget")

    print("### Evaluating Test Set ###")
    test_classification_result = evaluate_classification(parquet_file=args.test_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="test",
                                                                 forget_parquet_file=forget_parquet_file)

    test_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.test_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="test",
                                                                 forget_parquet_file=forget_parquet_file)

    test_generation_result = evaluate_generation(parquet_file=args.test_data,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="test",
                                                 forget_parquet_file=forget_parquet_file)

    print("### Evaluating Retain Shared Set ###")
    retain_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=retain_parquet_file,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="retain_shared")

    retain_classification_result = evaluate_classification(parquet_file=retain_parquet_file,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="retain_shared")

    retain_generation_result = evaluate_generation(parquet_file=retain_parquet_file,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="retain_shared")

    print("### Evaluating Real Celebrity Set ###")

    real_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.celebrity_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="retain_celebrity")

    real_classification_result = evaluate_classification(parquet_file=args.celebrity_data,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="retain_celebrity")

    real_generation_result = evaluate_generation(parquet_file=args.celebrity_data,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="retain_celebrity")

    # Output results
    print("Forget Set Results:")
    print(forget_classification_result)
    print(forget_generation_result)
    print(forget_fill_in_the_blank_result)

    print("Test Set Results:")
    print(test_fill_in_the_blank_result)
    print(test_classification_result)
    print(test_generation_result)

    print("Retain Set (shared dataset) Results:")
    print( retain_fill_in_the_blank_result)
    print(retain_classification_result)
    print(retain_generation_result)

    print("Retain Set (real person) Results:")
    print(real_fill_in_the_blank_result)
    print(real_classification_result)
    print(real_generation_result)

    output_file = f'{args.output_folder}/{args.output_file}_final_evaluation_results.json'
    # Prepare the data to be saved in JSON format
    results_data = {
        "Forget Set Results": {
            "fill_in_the_blank": forget_fill_in_the_blank_result,
            "classification": forget_classification_result,
            "generation": forget_generation_result
        },
        "Test Set Results": {
            "fill_in_the_blank": test_fill_in_the_blank_result,
            "classification": test_classification_result,
            "generation": test_generation_result,
        },
        "Retain Set (shared dataset) Results": {
            "fill_in_the_blank": retain_fill_in_the_blank_result,
            "classification": retain_classification_result,
            "generation": retain_generation_result
        },
        "Retain Set (real person) Results": {
            "fill_in_the_blank": real_fill_in_the_blank_result,
            "classification": real_classification_result,
            "generation": real_generation_result
        }
    }

    # Write the results to a local JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    # Optionally print a message to indicate successful save
    print(results_data)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
