from pyswip import Prolog
import os
import re
from typing import Dict, List, Optional

class TumorDiagnosisSystem:
    def __init__(self, facts_file: str, rules_file: str):
        """
        Initialize the diagnosis system with paths to facts and rules files.
        
        Args:
            facts_file: Path to the Prolog facts file
            rules_file: Path to the Prolog rules file
        """
        self.facts_file = os.path.abspath(facts_file).replace("\\", "/")
        self.rules_file = os.path.abspath(rules_file).replace("\\", "/")
        self.prolog = Prolog()
        self._validate_files()
        
    def _validate_files(self) -> None:
        """Validate that the input files exist and are readable."""
        if not os.path.exists(self.facts_file):
            raise FileNotFoundError(f"Facts file not found: {self.facts_file}")
        if not os.path.exists(self.rules_file):
            raise FileNotFoundError(f"Rules file not found: {self.rules_file}")
        if not os.access(self.facts_file, os.R_OK):
            raise PermissionError(f"Cannot read facts file: {self.facts_file}")
        if not os.access(self.rules_file, os.R_OK):
            raise PermissionError(f"Cannot read rules file: {self.rules_file}")
    
    def _consult_prolog_files(self) -> None:
        """Load both the rules and facts files into the Prolog engine."""
        try:
            # Consult the rules file first
            list(self.prolog.query(f"consult('{self.rules_file}')"))
            
            # Load and assert facts line by line
            with open(self.facts_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('%'):  # Skip empty lines and comments
                        if line.endswith('.'):
                            line = line[:-1]  # Remove trailing dot
                        try:
                            self.prolog.assertz(line)
                        except Exception as pe:  # Changed from PrologError to general Exception
                            print(f"Warning: Could not assert fact '{line}': {pe}")
        except Exception as e:
            raise RuntimeError(f"Error consulting Prolog files: {e}")
    
    def _get_tumor_ids(self) -> List[str]:
        """Extract all unique tumor IDs from the facts file."""
        tumor_ids = set()
        tumor_pattern = re.compile(r'tumor_\d+')
        
        with open(self.facts_file, 'r') as f:
            for line in f:
                matches = tumor_pattern.findall(line)
                tumor_ids.update(matches)
        
        return sorted(tumor_ids, key=lambda x: int(x.split('_')[1]))
    
    def _get_detailed_diagnosis(self, tumor_id: str) -> Dict:
        """
        Get comprehensive diagnosis information for a tumor.
        
        Args:
            tumor_id: The ID of the tumor to diagnose (e.g., 'tumor_1')
            
        Returns:
            Dictionary containing diagnosis details including:
            - primary_diagnosis: The main diagnosis
            - confidence: Confidence level if available
            - features: Relevant features contributing to diagnosis
            - differential: List of differential diagnoses
        """
        diagnosis_info = {
            'tumor_id': tumor_id,
            'primary_diagnosis': 'unknown',
            'confidence': None,
            'features': [],
            'differential': []
        }
        
        try:
            # Get primary diagnosis
            results = list(self.prolog.query(f"diagnosis({tumor_id}, Diagnosis)"))
            if results:
                diagnosis_info['primary_diagnosis'] = results[0]['Diagnosis']
            
            # Get confidence if available
            try:
                conf_results = list(self.prolog.query(f"diagnosis_confidence({tumor_id}, Confidence)"))
                if conf_results:
                    diagnosis_info['confidence'] = conf_results[0]['Confidence']
            except:
                pass
            
            # Get contributing features
            try:
                feature_results = list(self.prolog.query(f"contributing_features({tumor_id}, Features)"))
                if feature_results:
                    diagnosis_info['features'] = feature_results[0]['Features']
            except:
                pass
            
            # Get differential diagnoses
            try:
                diff_results = list(self.prolog.query(f"differential_diagnoses({tumor_id}, Differentials)"))
                if diff_results:
                    diagnosis_info['differential'] = diff_results[0]['Differentials']
            except:
                pass
            
        except Exception as pe:  # Changed from PrologError to general Exception
            print(f"Error querying for tumor {tumor_id}: {pe}")
            diagnosis_info['error'] = str(pe)
        
        return diagnosis_info
    
    def run_diagnosis(self) -> Dict[str, Dict]:
        """
        Run comprehensive diagnosis for all tumors found in the facts file.
        
        Returns:
            Dictionary mapping tumor IDs to their diagnosis information
        """
        self._consult_prolog_files()
        tumor_ids = self._get_tumor_ids()
        
        diagnoses = {}
        for tumor_id in tumor_ids:
            diagnoses[tumor_id] = self._get_detailed_diagnosis(tumor_id)
        
        return diagnoses
    
    def generate_report(self, diagnoses: Dict[str, Dict]) -> str:
        """
        Generate a human-readable report from the diagnosis results.
        
        Args:
            diagnoses: Dictionary of diagnosis results from run_diagnosis()
            
        Returns:
            Formatted multi-line string report
        """
        report = []
        report.append("TUMOR DIAGNOSIS REPORT")
        report.append("=" * 40)
        
        for tumor_id, info in diagnoses.items():
            report.append(f"\nTumor ID: {tumor_id}")
            report.append(f"Primary Diagnosis: {info['primary_diagnosis']}")
            
            if info['confidence']:
                report.append(f"Confidence: {info['confidence']}")
            
            if info['features']:
                features = ", ".join(info['features']) if isinstance(info['features'], list) else info['features']
                report.append(f"Key Features: {features}")
            
            if info['differential']:
                differentials = ", ".join(info['differential']) if isinstance(info['differential'], list) else info['differential']
                report.append(f"Differential Diagnoses: {differentials}")
            
            if 'error' in info:
                report.append(f"Error: {info['error']}")
        
        report.append("\n" + "=" * 40)
        report.append("END OF REPORT")
        
        return "\n".join(report)


if __name__ == "__main__":
    # File paths (edit these paths as per your system)
    FACTS_FILE = r"D:\Sajin Python Works\Sajin Work\Symbolic reasoning\sajin CT MRI\sajin CT MRI\tumor_concepts_facts.pl"
    RULES_FILE = r"D:\Sajin Python Works\Sajin Work\Symbolic reasoning\sajin CT MRI\sajin CT MRI\rules.pl"
    
    try:
        print("Initializing tumor diagnosis system...")
        diagnosis_system = TumorDiagnosisSystem(FACTS_FILE, RULES_FILE)
        
        print("Running diagnosis...")
        diagnoses = diagnosis_system.run_diagnosis()
        
        print("\nDiagnosis Results:")
        print(diagnosis_system.generate_report(diagnoses))
        
    except Exception as e:
        print(f"Error in tumor diagnosis system: {e}")