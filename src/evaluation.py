import json
import os
import re
from typing import Dict, List, Set
from comet import download_model, load_from_checkpoint
from config import REFERENCES_PATH, PREDICTION_PATH, TARGET_LANGUAGES

VERBOSE = False
ENTITY_TYPES = [
    "Musical work", "Artwork", "Food", "Animal", "Plant", "Book", "Book series",
    "Fictional entity", "Landmark", "Movie", "Place of worship", "Natural place",
    "TV series", "Person",
]

#COMET model
COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
NUM_GPUS = 1
BATCH_SIZE = 32

#Paths
MODEL_NAME = "mt5-small-noarabic"
SPLIT = "validation"


#These functions are from SemEval
def load_references(input_path: str, entity_types: List[str]) -> List[dict]:
    """
    Load data from the input file (JSONL) and return a list of dictionaries, one for each instance in the dataset.

    Args:
        input_path (str): Path to the input file.
        entity_types (List[str]): List of entity types to filter the evaluation.

    Returns:
        List[dict]: List of dictionaries, one for each instance in the dataset.
    """
    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_data = json.loads(line)

            # Skip instances with empty target list and log a warning.
            if not line_data["targets"]:
                print(f"Empty target list for instance {line_data['id']}")
                continue

            # Filter the evaluation to the specified entity types if provided.
            if entity_types and not any(
                e in line_data["entity_types"] for e in entity_types
            ):
                continue

            data.append(line_data)

    return data

def load_predictions(input_path: str) -> Dict[str, str]:
    """
    Load data from the input file (JSONL) and return a dictionary with the instance ID as key and the prediction as value.

    Args:
        input_path (str): Path to the input file.

    Returns:
        Dict[str, str]: Dictionary with the instance ID as key and the prediction as value.
    """
    data = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_data = json.loads(line)
            prediction = line_data["prediction"]

            # Get the instance ID from a substring of the ID.
            pattern = re.compile(r"Q[0-9]+_[0-9]")
            match = pattern.match(line_data["id"])
            if not match:
                raise ValueError(f"Invalid instance ID: {line_data['id']}")

            instance_id = match.group(0)
            data[instance_id] = prediction

    return data

def get_mentions_from_references(data: List[dict]) -> Dict[str, Set[str]]:
    """
    Load the ground truth entity mentions from the data.

    Args:
        data (List[dict]): List of dictionaries, one for each instance in the dataset.

    Returns:
        Dict[str, Set[str]]: Dictionary with the instance ID as key and the set of entity mentions as value.
    """
    mentions = {}

    for instance in data:
        instance_id = instance["id"]
        instance_mentions = set()

        for target in instance["targets"]:
            mention = target["mention"]
            instance_mentions.add(mention)

        mentions[instance_id] = instance_mentions

    return mentions

def compute_entity_name_translation_accuracy(
    predictions: Dict[str, str],
    mentions: Dict[str, Set[str]],
    verbose: bool = False,
) -> dict:
    """
    Compute the entity name translation accuracy.

    Args:
        predictions (Dict[str, str]): Predictions of the model.
        mentions (Dict[str, Set[str]]): Ground truth entity mentions.
        verbose (bool): Set to True to print every wrong match.

    Returns:
        dict: Dictionary with the following
            - correct: Number of correct matches.
            - total: Total number of instances.
            - accuracy: Accuracy of the model.
    """
    correct, total = 0, 0

    for instance_id, instance_mentions in mentions.items():
        # Check that there is at least one entity mention for the instance.
        assert instance_mentions, f"No mentions for instance {instance_id}"

        # Increment the total count of instances (for recall calculation).
        total += 1

        # Check that there is a prediction for the instance.
        if instance_id not in predictions:
            if verbose:
                print(
                    f"No prediction for instance {instance_id}. Check that this is expected behavior, as it may affect the evaluation."
                )
            continue

        prediction = predictions[instance_id]
        normalized_translation = prediction.casefold()
        entity_match = False

        for mention in instance_mentions:
            normalized_mention = mention.casefold()

            # Check if the normalized mention is a substring of the normalized translation.
            # If it is, consider the prediction (the entity name translation) correct.
            if normalized_mention in normalized_translation:
                correct += 1
                entity_match = True
                break

        # Log the prediction and the ground truth mentions for every wrong match if verbose is set.
        if not entity_match and verbose:
            print(f"Prediction: {prediction}")
            print(f"Ground truth mentions: {instance_mentions}")
            print("")

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
    }

def m_eta(PATH_TO_REFERENCES, PATH_TO_PREDICTIONS):
    # Load references
    print(f"Loading data from {PATH_TO_REFERENCES}...")
    reference_data = load_references(PATH_TO_REFERENCES, ENTITY_TYPES)
    mentions = get_mentions_from_references(reference_data)
    assert len(mentions) == len(reference_data)
    print(f"Loaded {len(reference_data)} instances.")

    # Load predictions
    print(f"Loading data from {PATH_TO_PREDICTIONS}...")
    prediction_data = load_predictions(PATH_TO_PREDICTIONS)
    print(f"Loaded {len(prediction_data)} predictions.")

    # Calculate M-ETA
    print("Computing entity name translation accuracy...")
    entity_name_translation_accuracy = compute_entity_name_translation_accuracy(
        prediction_data,
        mentions,
        verbose=VERBOSE,
    )

    return entity_name_translation_accuracy["accuracy"]

def comet(PATH_TO_REFERENCES, PATH_TO_PREDICTIONS, model):
    # Load references
    references = {}

    with open(PATH_TO_REFERENCES, "r", encoding="utf-8") as f:

        for line in f:
            data = json.loads(line)
            references[data["id"]] = data

    print(f"Loaded {len(references)} references from {PATH_TO_REFERENCES}")

    # Load predictions 
    predictions = {}

    with open(PATH_TO_PREDICTIONS, "r", encoding="utf-8") as f:

        for line in f:
            data = json.loads(line)
            predictions[data["id"]] = data

    print(f"Loaded {len(predictions)} predictions from {PATH_TO_PREDICTIONS}")

    # Get all those references that have a corresponding prediction
    ids = set(references.keys()) & set(predictions.keys())
    num_missing_predictions = len(references) - len(ids)

    if num_missing_predictions > 0:
        print(f"Missing predictions for {num_missing_predictions} references")
    else:
        print("All references have a corresponding prediction")

    # Create instances
    instance_ids = {}
    instances = []
    current_index = 0

    for id in sorted(list(ids)):
        reference = references[id]
        prediction = predictions[id]

        for target in reference["targets"]:
            instances.append(
                {
                    "src": reference["source"],
                    "ref": target["translation"],
                    "mt": prediction["prediction"],
                }
            )

        instance_ids[id] = [current_index, current_index + len(reference["targets"])]
        current_index += len(reference["targets"])

    print(f"Created {len(instances)} instances")

    # Compute the scores
    outputs = model.predict(instances, batch_size=BATCH_SIZE, gpus=NUM_GPUS)

    # Extract the scores
    scores = outputs.scores
    max_scores = []

    for id, indices in instance_ids.items():
        # Get the max score for each reference
        max_score = max(scores[indices[0] : indices[1]])
        max_scores.append(max_score)

    # Compute the average score while taking into account the missing predictions (which are considered as 0)
    system_score = sum(max_scores) / (len(max_scores) + num_missing_predictions)

    return system_score

def main():
    
    #Init COMET model
    model_path = download_model(COMET_MODEL_NAME)
    model = load_from_checkpoint(model_path)

    results = {}

    for target_language in TARGET_LANGUAGES:
        print(f"\nEvaluating {target_language}...")

        PATH_TO_REFERENCES = os.path.join(
            REFERENCES_PATH, SPLIT, f"{target_language}.jsonl"
        )

        PATH_TO_PREDICTIONS = os.path.join(
            PREDICTION_PATH, MODEL_NAME, SPLIT, f"{target_language}.jsonl"
        )

        #Compute COMET
        print("Computing COMET...")
        comet_score = comet(PATH_TO_REFERENCES, PATH_TO_PREDICTIONS, model) * 100.0
        print(f"COMET: {comet_score:.2f}")

        #Compute M-ETA
        print("Computing M-ETA...")
        m_eta_score = m_eta(PATH_TO_REFERENCES, PATH_TO_PREDICTIONS) * 100.0
        print(f"M_ETA: {m_eta_score:.2f}")

        results[target_language] = {
            "m_eta": m_eta_score,
            "comet": comet_score
        }



    print("\nFinal Results:")
    print("=" * 50)
    print(f"{'Language':<10} | {'M-ETA':<10} | {'COMET':<10}")
    print("-" * 50)

    # Compute averages
    total_m_eta = 0.0
    total_comet = 0.0
    for lang, scores in results.items():
        print(f"{lang:<10} | {scores['m_eta']:<10.2f} | {scores['comet']:<10.2f}")
        total_m_eta += scores["m_eta"]
        total_comet += scores["comet"]

    # Print averages
    avg_m_eta = total_m_eta / len(TARGET_LANGUAGES)
    avg_comet = total_comet / len(TARGET_LANGUAGES)
    print("-" * 50)
    print(f"{'Average':<10} | {avg_m_eta:<10.2f} | {avg_comet:<10.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()




