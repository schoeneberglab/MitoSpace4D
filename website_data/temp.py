import os
import os.path as osp
import json


if __name__ == '__main__':
    # Function to load colors
    def load_colors(filepath):
        color_dict = {}
        with open(filepath, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 6:
                    _, _, index, r, g, b = parts
                    color = [float(v) / 255 if float(v) > 1.0 else float(v) for v in (r, g, b)]
                    color_dict[int(index)] = color
        return color_dict


    drugs_to_labels = {}
    label_drug_dict = {}
    with open("/home/dhruvagarwal/projects/MitoSpace4D/extraction_utils/drugs_to_labels.txt", 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drugs_to_labels[drug] = int(label)
            label_drug_dict[int(label)] = drug



    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"
    colors = load_colors(osp.join(proj_dir, "extraction_utils/colors.txt"))
    colors_phenotypic = load_colors(osp.join(proj_dir, "extraction_utils/colors_phenotypic.txt"))

    # Load the JSON file
    json_file_path = "/home/dhruvagarwal/projects/MitoSpace4D/metadata/points4d.json"

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # replace the color and color_phenotypic values in the JSON file
    for point in data['points']:
        label = drugs_to_labels[point['phenotype']]
        if label in colors_phenotypic:
            r, g, b = colors_phenotypic[label]
            point['color_phenotypic'] = {"r": r, "g": g, "b": b}
        else:
            print(f"Label {label} not found in colors_phenotypic")

    # save the modified JSON file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)