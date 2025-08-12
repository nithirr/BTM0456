from pyswip import Prolog
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict
import tempfile
import os
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Global Times New Roman settings
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'font.size': 23,
    'axes.titlesize': 23,
    'axes.labelsize': 23,
    'xtick.labelsize': 23,
    'ytick.labelsize': 23,
    'figure.titlesize': 23,
    'legend.fontsize': 22,
    'legend.title_fontsize': 22
})

class TumorDiagnosisVisualizer:
    def __init__(self):
        self.diagnosis_colors = {
            'glioma': '#FF6B6B',
            'meningioma': '#4ECDC4',
            'metastasis': '#FFE66D',
            'unknown': '#A5A5A5'
        }
        self.confidence_colors = {
            'high': '#2ECC71',
            'medium': '#F39C12',
            'low': '#E74C3C',
            'none': '#BDC3C7'
        }
        self.feature_colors = {
            'shape': '#3498DB',
            'margin': '#9B59B6',
            'effect': '#1ABC9C',
            'enhancement': '#E67E22',
            'other': '#95A5A6'
        }
        
        self.prolog = Prolog()
        self.graph = nx.DiGraph()
        self._load_prolog_knowledge()
        self._build_enhanced_visualization_graph()

    def _load_prolog_knowledge(self):
        facts = [
            "has_feature(tumor_1, irregular_shape)",
            "has_feature(tumor_1, infiltrative_margins)",
            "has_feature(tumor_1, mass_effect)",
            "has_feature(tumor_1, necrosis_present)",
            "has_feature(tumor_2, well_defined_margins)",
            "has_feature(tumor_2, dural_tail)",
            "has_feature(tumor_2, homogeneous_enhancement)",
            "has_feature(tumor_3, multiple_lesions)",
            "has_feature(tumor_3, vasogenic_edema)",
            "has_feature(tumor_4, irregular_shape)",
            "has_feature(tumor_4, dural_tail)"
        ]
        for fact in facts:
            self.prolog.assertz(fact)

        rules = """
diagnosis(TumorID, glioma) :- 
    has_feature(TumorID, irregular_shape),
    has_feature(TumorID, infiltrative_margins),
    has_feature(TumorID, mass_effect).
    
diagnosis(TumorID, meningioma) :- 
    has_feature(TumorID, well_defined_margins),
    has_feature(TumorID, dural_tail),
    has_feature(TumorID, homogeneous_enhancement).
    
diagnosis(TumorID, metastasis) :- 
    has_feature(TumorID, multiple_lesions),
    has_feature(TumorID, vasogenic_edema),
    has_feature(TumorID, known_primary).
    
diagnosis_confidence(TumorID, high) :- 
    diagnosis(TumorID, _),
    count_supporting_features(TumorID, Count),
    Count >= 3.
    
diagnosis_confidence(TumorID, medium) :- 
    diagnosis(TumorID, _),
    count_supporting_features(TumorID, Count),
    Count == 2.
    
diagnosis_confidence(TumorID, low) :- 
    diagnosis(TumorID, _),
    count_supporting_features(TumorID, Count),
    Count == 1.
    
count_supporting_features(TumorID, Count) :- 
    findall(Feature, has_feature(TumorID, Feature), Features),
    length(Features, Count).

feature_category(irregular_shape, shape).
feature_category(well_defined_margins, margin).
feature_category(infiltrative_margins, margin).
feature_category(mass_effect, effect).
feature_category(dural_tail, margin).
feature_category(homogeneous_enhancement, enhancement).
feature_category(multiple_lesions, other).
feature_category(vasogenic_edema, effect).
feature_category(known_primary, other).
feature_category(necrosis_present, other).
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
            f.write(rules)
            temp_path = f.name

        try:
            consult_path = temp_path.replace("\\", "/")
            list(self.prolog.query(f"consult('{consult_path}')"))
        finally:
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temporary file: {cleanup_error}")

    def _build_enhanced_visualization_graph(self):
        for diagnosis, color in self.diagnosis_colors.items():
            if diagnosis != 'unknown':
                self.graph.add_node(
                    diagnosis.capitalize(),
                    type="diagnosis",
                    color=color,
                    size=3000,
                    font_size=24,
                    shape='s'
                )
        features = set()
        for result in self.prolog.query("has_feature(_, Feature)"):
            features.add(result["Feature"])
            
        for feature in features:
            category = next(self.prolog.query(f"feature_category({feature}, Category)"))["Category"]
            color = self.feature_colors.get(category, self.feature_colors['other'])
            self.graph.add_node(
                feature,
                type="feature",
                color=color,
                size=2000,
                font_size=22,
                shape='o'
            )

        diagnosis_feature_map = {
            'Glioma': ['irregular_shape', 'infiltrative_margins', 'mass_effect'],
            'Meningioma': ['well_defined_margins', 'dural_tail', 'homogeneous_enhancement'],
            'Metastasis': ['multiple_lesions', 'vasogenic_edema', 'known_primary']
        }
        for diagnosis, feature_list in diagnosis_feature_map.items():
            for feature in feature_list:
                if feature in self.graph.nodes:
                    category = next(self.prolog.query(f"feature_category({feature}, Category)"))["Category"]
                    weight = 3 if category in ['shape', 'margin'] else 2 if category == 'effect' else 1
                    self.graph.add_edge(
                        feature,
                        diagnosis,
                        weight=weight,
                        color='#7F8C8D',
                        alpha=0.7
                    )

    def visualize_knowledge_graph(self, layout='spring'):
        plt.figure(figsize=(16, 12))
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1.5, seed=42, weight='weight')
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        node_colors = [self.graph.nodes[n]['color'] for n in self.graph.nodes]
        node_sizes = [self.graph.nodes[n]['size'] for n in self.graph.nodes]
        node_shapes = [self.graph.nodes[n]['shape'] for n in self.graph.nodes]
        
        for shape in set(node_shapes):
            nodes = [n for n in self.graph.nodes if self.graph.nodes[n]['shape'] == shape]
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=nodes,
                node_shape=shape,
                node_color=[self.graph.nodes[n]['color'] for n in nodes],
                node_size=[self.graph.nodes[n]['size'] for n in nodes],
                alpha=0.9,
                edgecolors='black',
                linewidths=2
            )
        
        edges = self.graph.edges()
        weights = [self.graph[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            self.graph, pos,
            width=[w * 0.8 for w in weights],
            alpha=0.6,
            edge_color='#34495E',
            arrowstyle='-|>',
            arrowsize=15
        )
        
        for node in self.graph.nodes():
            nx.draw_networkx_labels(
                self.graph, {node: pos[node]},
                labels={node: node.replace('_', ' ').title()},
                font_size=self.graph.nodes[node]['font_size'],
                font_family='Times New Roman',
                font_weight='bold'
            )
        
        legend_elements = [
            Patch(facecolor=self.diagnosis_colors['glioma'], edgecolor='black', label='Glioma Diagnosis'),
            Patch(facecolor=self.diagnosis_colors['meningioma'], edgecolor='black', label='Meningioma Diagnosis'),
            Patch(facecolor=self.diagnosis_colors['metastasis'], edgecolor='black', label='Metastasis Diagnosis'),
            Patch(facecolor=self.feature_colors['shape'], edgecolor='black', label='Shape Features'),
            Patch(facecolor=self.feature_colors['margin'], edgecolor='black', label='Margin Features'),
            Patch(facecolor=self.feature_colors['effect'], edgecolor='black', label='Effect Features'),
            Patch(facecolor=self.feature_colors['enhancement'], edgecolor='black', label='Enhancement Features'),
            Patch(facecolor=self.feature_colors['other'], edgecolor='black', label='Other Features'),
            Line2D([0], [0], color='#34495E', lw=4, label='Strong Correlation'),
            Line2D([0], [0], color='#34495E', lw=2, label='Medium Correlation'),
            Line2D([0], [0], color='#34495E', lw=1, label='Weak Correlation')
        ]
        
        plt.legend(
            handles=legend_elements,
            loc='center',
            title="Graph Legend",
            framealpha=0.2,
            prop={'family': 'Times New Roman', 'size': 22},
            title_fontproperties={'family': 'Times New Roman', 'weight': 'bold', 'size': 22}
        )
        
       
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def run_diagnosis(self) -> Dict[str, Dict]:
        tumor_ids = list(set(res['TumorID'] for res in self.prolog.query("has_feature(TumorID, _)")))
        diagnoses = {}
        for tumor_id in tumor_ids:
            info = {
                'tumor_id': tumor_id,
                'possible_diagnoses': [],
                'primary_diagnosis': 'unknown',
                'confidence': 'none',
                'features': [],
                'feature_categories': {}
            }
            dx_results = list(self.prolog.query(f"diagnosis({tumor_id}, Diagnosis)"))
            if dx_results:
                info['possible_diagnoses'] = [res['Diagnosis'] for res in dx_results]
                info['primary_diagnosis'] = dx_results[0]['Diagnosis']
            conf_results = list(self.prolog.query(f"diagnosis_confidence({tumor_id}, Confidence)"))
            if conf_results:
                info['confidence'] = conf_results[0]['Confidence']
            features = list(self.prolog.query(f"has_feature({tumor_id}, Feature)"))
            info['features'] = [f['Feature'] for f in features]
            for feature in info['features']:
                category = list(self.prolog.query(f"feature_category({feature}, Category)"))[0]['Category']
                info['feature_categories'][feature] = category
            diagnoses[tumor_id] = info
        return diagnoses

    def visualize_diagnosis_results(self, diagnoses: Dict[str, Dict]):
        plt.figure(figsize=(14, 8))
        tumors = list(diagnoses.keys())
        diagnoses_list = [d['primary_diagnosis'] for d in diagnoses.values()]
        confidences = [d['confidence'] for d in diagnoses.values()]
        feature_counts = [len(d['features']) for d in diagnoses.values()]
        diag_colors = [self.diagnosis_colors.get(d, self.diagnosis_colors['unknown']) for d in diagnoses_list]
        conf_colors = [self.confidence_colors.get(c, self.confidence_colors['none']) for c in confidences]
        
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
        ax2 = plt.subplot2grid((3, 3), (2, 0))
        ax3 = plt.subplot2grid((3, 3), (2, 1))
        ax4 = plt.subplot2grid((3, 3), (2, 2))
        
        bars = ax1.bar(
            x=tumors,
            height=feature_counts,
            color=diag_colors,
            edgecolor='black',
            alpha=0.8
        )
        for bar, conf_color in zip(bars, conf_colors):
            bar.set_edgecolor(conf_color)
            bar.set_linewidth(3)
        
        ax1.set_title('Tumor Diagnoses with Confidence Levels', fontsize=24, fontname='Times New Roman')
        ax1.set_ylabel('Number of Features Detected', fontname='Times New Roman')
        ax1.set_xticks(range(len(tumors)))
        ax1.set_xticklabels(tumors, rotation=45, ha='right', fontname='Times New Roman')
        
        unique_diag, counts = np.unique(diagnoses_list, return_counts=True)
        ax2.pie(
            counts,
            labels=unique_diag,
            colors=[self.diagnosis_colors[d] for d in unique_diag],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('Diagnosis Distribution', fontname='Times New Roman')
        
        unique_conf, conf_counts = np.unique(confidences, return_counts=True)
        ax3.pie(
            conf_counts,
            labels=unique_conf,
            colors=[self.confidence_colors[c] for c in unique_conf],
            autopct='%1.1f%%',
            startangle=90
        )
        ax3.set_title('Confidence Level Distribution', fontname='Times New Roman')
        
        if diagnoses:
            first_tumor = next(iter(diagnoses.values()))
            categories = list(first_tumor['feature_categories'].values())
            unique_cat, cat_counts = np.unique(categories, return_counts=True)
            ax4.bar(
                x=range(len(unique_cat)),
                height=cat_counts,
                color=[self.feature_colors[c] for c in unique_cat],
                tick_label=unique_cat
            )
            ax4.set_title(f"Feature Categories for {first_tumor['tumor_id']}", fontname='Times New Roman')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Initializing Enhanced Tumor Diagnosis Visualizer...")
    visualizer = TumorDiagnosisVisualizer()
    print("\nVisualizing knowledge graph (spring layout)...")
    visualizer.visualize_knowledge_graph(layout='spring')
    print("\nVisualizing knowledge graph (circular layout)...")
    visualizer.visualize_knowledge_graph(layout='circular')
    print("\nRunning diagnosis...")
    diagnoses = visualizer.run_diagnosis()
    print("\nDiagnosis Results:")
    for tumor_id, info in diagnoses.items():
        print(f"\nTumor {tumor_id}:")
        print(f"  Primary Diagnosis: {info['primary_diagnosis']}")
        print(f"  Confidence: {info['confidence']}")
        print(f"  Features ({len(info['features'])}):")
        for feature, category in info['feature_categories'].items():
            print(f"    - {feature.replace('_', ' ').title()} ({category})")
    print("\nVisualizing diagnosis results...")
    visualizer.visualize_diagnosis_results(diagnoses)
