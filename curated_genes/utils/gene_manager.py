import logging
import json
from typing import List, Tuple, Optional, Dict, Set


class GeneManager:
    def __init__(self, gene_file):
        self.genes = self.load_genes(gene_file)

    def load_genes(self, gene_file):
        with open(gene_file, 'r') as f:
            return json.load(f)

    def read_fasta_file_genes(self, file_path, target, filter_xs=False, sort_by_genes=True,
                              use_gene_file=True, method="separate"):
        if use_gene_file:
            sorted_genes = {key: sorted(value) for key, value in self.genes.items()}
            gene_list = [gene.lower() for gene in sorted_genes[target]]
            gene_sequences = {gene: {'sequence': None, 'before_ir': None, 'after_ir': None}
                              for gene in gene_list}
        else:
            gene_list = None
            sorted_genes = set()

            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.startswith('>'):
                            gene_name = line.split("|")[-3].strip("]").lower()
                            if "ir" not in gene_name:
                                sorted_genes.add(gene_name)
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                return [], []

            sorted_genes = sorted(sorted_genes)
            gene_sequences = {}

        file_gene_order = []
        temp_dict = {'before_ir': None, 'sequence': None, 'after_ir': None, 'gene_name': None}

        try:
            with open(file_path, 'r') as file:
                sequence = ''
                current_header = None

                def process_temp_dict():
                    if temp_dict['gene_name']:
                        gene_name = temp_dict['gene_name']
                        if gene_name not in gene_sequences:
                            gene_sequences[gene_name] = {'sequence': None, 'before_ir': None, 'after_ir': None}
                            if gene_name not in file_gene_order:
                                file_gene_order.append(gene_name)
                        if temp_dict['sequence']:
                            gene_sequences[gene_name]['sequence'] = temp_dict['sequence']
                        if temp_dict['before_ir']:
                            gene_sequences[gene_name]['before_ir'] = temp_dict['before_ir']
                        if temp_dict['after_ir']:
                            gene_sequences[gene_name]['after_ir'] = temp_dict['after_ir']

                    temp_dict['before_ir'] = None
                    temp_dict['sequence'] = None
                    temp_dict['after_ir'] = None
                    temp_dict['gene_name'] = None

                for line in file:
                    if "<<P>>" in line:
                        return None

                    line = line.strip()
                    if line.startswith('>'):
                        if sequence:
                            clean_sequence = sequence.replace('\n', '').upper()

                            if current_header and 'IR:' in current_header:
                                if '|BEFORE|' in current_header or "before" in current_header:
                                    if temp_dict['before_ir'] or temp_dict['sequence']:
                                        process_temp_dict()
                                    temp_dict['before_ir'] = clean_sequence
                                elif '|AFTER|' in current_header or "after" in current_header:
                                    temp_dict['after_ir'] = clean_sequence
                                    process_temp_dict()
                            else:
                                gene_name = current_header.split("|")[-3].strip("]").lower()
                                if (use_gene_file and gene_name in gene_list) or not use_gene_file:
                                    if temp_dict['sequence']:
                                        process_temp_dict()
                                    temp_dict['sequence'] = clean_sequence
                                    temp_dict['gene_name'] = gene_name

                        sequence = ''
                        current_header = line
                    else:
                        sequence += line.upper()

                # Process the last sequence
                if sequence:
                    clean_sequence = sequence.replace('\n', '').upper()
                    if current_header and 'IR:' in current_header:
                        if '|BEFORE|' in current_header or "before" in current_header:
                            temp_dict['before_ir'] = clean_sequence
                        elif '|AFTER|' in current_header or "after" in current_header:
                            temp_dict['after_ir'] = clean_sequence
                    else:
                        gene_name = current_header.split("|")[-3].strip("]").lower()
                        if (use_gene_file and gene_name in gene_list) or not use_gene_file:
                            temp_dict['sequence'] = clean_sequence
                            temp_dict['gene_name'] = gene_name

                    process_temp_dict()

        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None

        # Choose which gene order to use
        if use_gene_file:
            genes_to_process = sorted_genes[target] if sort_by_genes else file_gene_order
        else:
            genes_to_process = sorted_genes

        combined_sequences = []
        genes_in_isolate = []
        missing = []

        for gene in genes_to_process:
            gene_lower = gene.lower()
            gene_data = gene_sequences.get(gene_lower)

            if gene_data:
                sequence = gene_data['sequence']
                ir_before = gene_data['before_ir']
                ir_after = gene_data['after_ir']

                is_missing = (sequence is None or "X" in sequence)

                if method == "no_intergenic":
                    if is_missing:
                        combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    else:
                        combined_sequences.append(sequence)
                    genes_in_isolate.append(gene)

                elif method == "merged":
                    if ir_before is None:
                        ir_before = "YYYYYYYYYYYY"
                    if ir_after is None:
                        ir_after = "YYYYYYYYYYYY"
                    if is_missing:
                        combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    else:
                        combined_sequences.append(ir_before + sequence + ir_after)
                    genes_in_isolate.append(gene)

                else:  # method == "separate"
                    if ir_before is None:
                        ir_before = "YYYYYYYYYYYY"
                    combined_sequences.append(ir_before)

                    if is_missing:
                        combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    else:
                        combined_sequences.append(sequence)

                    if ir_after is None:
                        ir_after = "YYYYYYYYYYYY"
                    combined_sequences.append(ir_after)

                    genes_in_isolate.append(gene + "_ir_before")
                    genes_in_isolate.append(gene)
                    genes_in_isolate.append(gene + "_ir_after")

                if is_missing:
                    missing.append(gene)

            else:
                # Gene is missing entirely
                missing.append(gene)
                if method == "no_intergenic":
                    combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    genes_in_isolate.append(gene)
                elif method == "merged":
                    combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    genes_in_isolate.append(gene)
                else:  # separate
                    combined_sequences.extend(["YYYYYYYYYYYY", "XXXXXXXXXXXXXXXXXX", "YYYYYYYYYYYY"])
                    genes_in_isolate.append(gene + "_ir_before")
                    genes_in_isolate.append(gene)
                    genes_in_isolate.append(gene + "_ir_after")

        if filter_xs and method == "separate":
            filtered_sequences = []
            filtered_genes = []
            for i in range(0, len(combined_sequences), 3):
                if "X" not in combined_sequences[i + 1]:
                    filtered_sequences.extend(combined_sequences[i:i + 3])
                    filtered_genes.extend(genes_in_isolate[i:i + 3])
            combined_sequences = filtered_sequences
            genes_in_isolate = filtered_genes
        elif filter_xs and method in ["no_intergenic", "merged"]:
            filtered_sequences = []
            filtered_genes = []
            for seq, g in zip(combined_sequences, genes_in_isolate):
                if "X" not in seq:
                    filtered_sequences.append(seq)
                    filtered_genes.append(g)
            combined_sequences = filtered_sequences
            genes_in_isolate = filtered_genes

        if not combined_sequences:
            logging.warning(f"No valid sequences found in file {file_path}")

        return combined_sequences, genes_in_isolate

    def read_fasta_file_genes_all(self, file_path, target=None, filter_xs=False):
        exclude_hypothetical = False
        filter_xs = False

        sorted_genes = set()

        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('>'):
                        gene_name = line.split("|")[-3].strip("]").lower()
                        if exclude_hypothetical and "hypothetical" in gene_name:
                            continue
                        sorted_genes.add(gene_name)
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return [], []

        sorted_genes = sorted(sorted_genes)
        gene_sequences = {gene: None for gene in sorted_genes}

        try:
            with open(file_path, 'r') as file:
                sequence = ''
                gene_name = None
                for line in file:
                    line = line.strip()
                    if line.startswith('>'):
                        if sequence and gene_name and gene_name in gene_sequences:
                            gene_sequences[gene_name] = sequence.upper()
                        sequence = ''
                        gene_name = line.split("|")[-3].strip("]").lower()
                        if exclude_hypothetical and "hypothetical" in gene_name:
                            gene_name = None
                    else:
                        sequence += line.upper()

                if sequence and gene_name and gene_name in gene_sequences:
                    gene_sequences[gene_name] = sequence.upper()

        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

        ordered_sequences = []
        genes_in_isolate = []
        missing = []

        for gene in sorted_genes:
            sequence = gene_sequences.get(gene, None)
            if sequence is None or "X" in sequence:
                missing.append(gene)
                ordered_sequences.append("XXXXXXXXXXXXXXXXXX")
            else:
                ordered_sequences.append(sequence)
            genes_in_isolate.append(gene)

        if filter_xs:
            filtered_sequences = [seq for seq in ordered_sequences if "X" not in seq]
            filtered_genes = [gene for seq, gene in zip(ordered_sequences, genes_in_isolate) if "X" not in seq]
            ordered_sequences = filtered_sequences
            genes_in_isolate = filtered_genes

        if not ordered_sequences:
            logging.warning(f"No valid sequences found in file {file_path}")

        if len(ordered_sequences) < 505:
            while len(ordered_sequences) < 505:
                ordered_sequences += "XXXXXXXX"
            ordered_sequences = ordered_sequences[:505]

        return ordered_sequences, genes_in_isolate