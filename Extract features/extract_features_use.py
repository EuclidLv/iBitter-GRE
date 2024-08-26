import os
import csv
import sys

import numpy as np
import torch
from Bio import SeqIO
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import EsmTokenizer, EsmModel



model_name = "facebook/esm2_t6_8M_UR50D"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)



amino_acid_weights = {
    'A': 89.09, 'C': 121.15, 'D': 133.1, 'E': 147.13, 'F': 165.19,
    'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.2,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
}

hydrophobicity_index = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

polarity = {
    'A': 'nonpolar', 'C': 'polar', 'D': 'polar', 'E': 'polar', 'F': 'nonpolar',
    'G': 'nonpolar', 'H': 'polar', 'I': 'nonpolar', 'K': 'polar', 'L': 'nonpolar',
    'M': 'nonpolar', 'N': 'polar', 'P': 'nonpolar', 'Q': 'polar', 'R': 'polar',
    'S': 'polar', 'T': 'polar', 'V': 'nonpolar', 'W': 'nonpolar', 'Y': 'polar'
}

def calculate_molecular_weight(sequence):
    return sum([amino_acid_weights[aa] for aa in sequence])

def calculate_hydrophobicity(sequence):
    return np.mean([hydrophobicity_index[aa] for aa in sequence])

def calculate_polarity(sequence):
    polar_count = sum([1 for aa in sequence if polarity[aa] == 'polar'])
    nonpolar_count = len(sequence) - polar_count
    return polar_count / len(sequence), nonpolar_count / len(sequence)

def calculate_isoelectric_point(sequence):
    analyser = ProteinAnalysis(sequence)
    return analyser.isoelectric_point()

def calculate_amino_acid_composition(sequence):
    composition = Counter(sequence)
    total = sum(composition.values())
    return {aa: count / total for aa, count in composition.items()}

def calculate_transition_frequency(sequence):
    transitions = Counter()
    for i in range(len(sequence) - 1):
        pair = (sequence[i], sequence[i + 1])
        transitions[pair] += 1
    total_transitions = sum(transitions.values())
    return {pair: count / total_transitions for pair, count in transitions.items()}

def calculate_amino_acid_distribution(sequence):
    length = len(sequence)
    distribution = {}
    for aa in set(sequence):
        indices = [i for i, x in enumerate(sequence) if x == aa]
        distribution[aa] = {
            'first_25%': sum(1 for i in indices if i < length * 0.25) / length,
            'middle_50%': sum(1 for i in indices if length * 0.25 <= i < length * 0.75) / length,
            'last_25%': sum(1 for i in indices if i >= length * 0.75) / length
        }
    return distribution

def extract_esm2_features(sequences):
    features = []
    for sequence in sequences:
        inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs)
        feature = outputs.last_hidden_state.mean(dim=1).squeeze()
        features.append(feature.cpu().numpy())
    return np.array(features)

def extract_features(sequence):
    features = []
    feature_names = []

    features.append(calculate_molecular_weight(sequence))
    feature_names.append('molecular_weight')

    features.append(calculate_hydrophobicity(sequence))
    feature_names.append('hydrophobicity')

    polar, nonpolar = calculate_polarity(sequence)
    features.append(polar)
    feature_names.append('polar_count')
    features.append(nonpolar)
    feature_names.append('nonpolar_count')

    features.append(calculate_isoelectric_point(sequence))
    feature_names.append('isoelectric_point')

    composition = calculate_amino_acid_composition(sequence)
    for aa in amino_acid_weights.keys():
        features.append(composition.get(aa, 0))
        feature_names.append(f'composition_{aa}')

    transition_frequency = calculate_transition_frequency(sequence)
    for aa1 in amino_acid_weights.keys():
        for aa2 in amino_acid_weights.keys():
            features.append(transition_frequency.get((aa1, aa2), 0))
            feature_names.append(f'transition_{aa1}_{aa2}')

    distribution = calculate_amino_acid_distribution(sequence)
    for aa in amino_acid_weights.keys():
        features.extend([
            distribution.get(aa, {}).get('first_25%', 0),
            distribution.get(aa, {}).get('middle_50%', 0),
            distribution.get(aa, {}).get('last_25%', 0)
        ])
        feature_names.extend([
            f'distribution_{aa}_first_25%',
            f'distribution_{aa}_middle_50%',
            f'distribution_{aa}_last_25%'
        ])

    return features, feature_names

def extract_sequence_and_descriptions(fasta_file_path):
    sequences, descriptions = [], []

    for record in SeqIO.parse(fasta_file_path, "fasta"):
        sequence = str(record.seq).upper()
        sequences.append(sequence)

        index = record.description.split()[0] 
        description = f"{index}_{sequence}"
        descriptions.append(description)

    return sequences, descriptions

def load_dimension_selection():
    index_file_path = r'index.csv'                  #you should use ur own path

    if not os.path.exists(index_file_path):
        raise FileNotFoundError(f"Dimension selection file not found: {index_file_path}")

    with open(index_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        selected_features = [row[0] for row in reader]


    return selected_features


def process_fasta_file(filepath):
    sequences, descriptions = extract_sequence_and_descriptions(filepath)

    output_file = filepath.replace('uploads', 'processed') + '_features.csv'

    selected_features = load_dimension_selection()

    with open(output_file, 'w', newline='') as csvfile:
        writer = None

        for sequence, description in zip(sequences, descriptions):
            features, feature_names = extract_features(sequence)
            print(f"Available features: {feature_names}")

            selected_feature_values = [features[feature_names.index(f)] for f in selected_features]

            esm2_features = extract_esm2_features([sequence])
            esm2_feature_names = [f'ESM2_feature_{i}' for i in range(esm2_features.shape[1])]

            all_features = selected_feature_values + list(esm2_features[0])
            all_feature_names = selected_features + esm2_feature_names

            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=['description'] + all_feature_names)
                writer.writeheader()

            row = {'description': description}
            row.update(dict(zip(all_feature_names, all_features)))
            writer.writerow(row)

    return output_file

if __name__ == "__main__":
    filepath = sys.argv[1]
    result = process_fasta_file(filepath)
    if result.endswith('_features.csv'):                    #you should use ur own path
        print(result)
        sys.exit(0)
    else:
        print(result)
        sys.exit(1)
