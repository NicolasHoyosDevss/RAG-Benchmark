#!/usr/bin/env python3
"""
RAG Comparison Table Generator

Este archivo genera DataFrames de pandas comparando los 4 RAGs basándose en el archivo
JSON consolidado generado por el sistema RAGAS. Compara tanto métricas RAGAS 
como métricas de performance (tokens, costos, tiempos).

Utiliza únicamente el archivo consolidado (ragas_comparison_all_rags_*.json) que contiene
todos los RAGs evaluados.

Uso:
    python results/utils/rag_comparison_table.py
    
    # O desde otro script:
    from results.utils.rag_comparison_table import get_rag_comparison_dataframe, get_question_by_question_dataframe
    
    # DataFrame general comparativo
    df_general = get_rag_comparison_dataframe()
    print(df_general)
    
    # DataFrame detallado por pregunta
    df_questions = get_question_by_question_dataframe()
    print(df_questions)
"""

import json
import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class RAGComparisonTable:
    """Generador de DataFrame comparativo entre RAGs"""
    
    def __init__(self, results_dir: str = None):
        """
        Inicializar el generador de comparaciones.
        
        Args:
            results_dir (str): Directorio donde están los JSON de evaluación
        """
        if results_dir is None:
            # Si no se especifica, buscar automáticamente el directorio results
            current_dir = Path(__file__).parent
            if current_dir.name == "utils":
                # Si estamos en results/utils/, ir al directorio padre
                self.results_dir = current_dir.parent
            else:
                # Si estamos en otro lugar, buscar el directorio results
                self.results_dir = Path("results")
        else:
            self.results_dir = Path(results_dir)
        self.rag_types = ["simple", "hybrid", "hyde", "rewriter"]
        self.rag_names = {
            "simple": "Simple Semantic RAG",
            "hybrid": "Hybrid RAG (BM25 + Semantic)", 
            "hyde": "HyDE RAG (Hypothetical Documents)",
            "rewriter": "Rewriter RAG (Multi-Query)"
        }
    
    def find_latest_consolidated_file(self) -> Optional[Path]:
        """
        Buscar el archivo JSON consolidado más reciente.
        
        Returns:
            Path del archivo consolidado más reciente o None si no se encuentra
        """
        consolidated_pattern = "ragas_comparison_all_rags_*.json"
        print(f"🔍 Buscando en: {self.results_dir.absolute()}")
        consolidated_files = list(self.results_dir.glob(consolidated_pattern))
        
        if consolidated_files:
            # Ordenar por fecha en el nombre del archivo (más reciente primero)
            consolidated_files.sort(key=lambda x: x.stem.split('_')[-2:], reverse=True)
            latest_consolidated = consolidated_files[0]
            print(f"✅ ARCHIVO CONSOLIDADO: {latest_consolidated.name}")
            return latest_consolidated
        else:
            print("❌ No se encontró archivo consolidado de evaluación")
            print(f"   Patrón buscado: {consolidated_pattern}")
            print(f"   Archivos encontrados en {self.results_dir}: {list(self.results_dir.glob('*.json'))}")
            return None
    
    def load_evaluation_data(self, file_path: Path) -> Optional[Dict]:
        """
        Cargar datos de evaluación desde un archivo JSON.
        
        Args:
            file_path (Path): Ruta al archivo JSON
            
        Returns:
            Dict con los datos de evaluación o None si hay error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando {file_path}: {e}")
            return None
    
    def extract_metrics_from_consolidated(self, evaluation_data: Dict, rag_type: str) -> Dict:
        """
        Extraer métricas relevantes del archivo consolidado.
        
        Args:
            evaluation_data (Dict): Datos de evaluación del archivo consolidado
            rag_type (str): Tipo de RAG
            
        Returns:
            Dict con métricas extraídas y organizadas
        """
        if not evaluation_data or rag_type not in evaluation_data.get('summary', {}):
            return {}
        
        # Extraer metadatos
        metadata = evaluation_data.get('metadata', {})
        rag_summary = evaluation_data['summary'][rag_type]
        rag_metrics = rag_summary.get('metrics', {})
        rag_performance = rag_summary.get('performance', {})
        
        # Organizar métricas
        metrics = {
            # Información del sistema
            'rag_name': rag_summary.get('rag_name', 'Unknown'),
            'rag_type': rag_type,
            'dataset_size': metadata.get('dataset_size', 0),
            'timestamp': metadata.get('timestamp', ''),
            
            # Métricas RAGAS (0-1, mayor es mejor)
            'faithfulness': round(rag_metrics.get('faithfulness', 0), 4),
            'answer_relevancy': round(rag_metrics.get('answer_relevancy', 0), 4),
            'context_precision': round(rag_metrics.get('context_precision', 0), 4),
            'context_recall': round(rag_metrics.get('context_recall', 0), 4),
            'overall_average_score': round(rag_metrics.get('overall_average', 0), 4),
            
            # Métricas de Performance
            'avg_execution_time': rag_performance.get('avg_execution_time', 0),
            'total_input_tokens': rag_performance.get('total_input_tokens', 0),
            'total_output_tokens': rag_performance.get('total_output_tokens', 0),
            'total_cost': rag_performance.get('total_cost', 0),
            'avg_cost_per_question': rag_performance.get('avg_cost_per_question', 0),
            
            # Métricas derivadas
            'total_tokens': rag_performance.get('total_tokens', 0),
            'cost_per_token': rag_performance.get('cost_per_token', 0)
        }
        
        return metrics
    

    
    def generate_comparison_dataframe(self, consolidated_file: Path) -> pd.DataFrame:
        """
        Generar DataFrame comparativo entre los RAGs desde archivo consolidado.
        
        Args:
            consolidated_file (Path): Ruta al archivo JSON consolidado
            
        Returns:
            DataFrame de pandas con la comparación
        """
        # Cargar datos del archivo consolidado
        evaluation_data = self.load_evaluation_data(consolidated_file)
        if not evaluation_data:
            return pd.DataFrame()
        
        # Extraer métricas de todos los RAGs
        all_metrics = []
        
        for rag_type in self.rag_types:
            metrics = self.extract_metrics_from_consolidated(evaluation_data, rag_type)
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return pd.DataFrame()  # DataFrame vacío si no hay datos
        
        # Crear DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Ordenar por overall_average_score descendente
        df = df.sort_values('overall_average_score', ascending=False).reset_index(drop=True)
        
        return df
    
    def generate_question_by_question_dataframe(self, consolidated_file: Path) -> pd.DataFrame:
        """
        Generar DataFrame detallado con resultados por pregunta desde archivo consolidado.
        
        Args:
            consolidated_file (Path): Ruta al archivo JSON consolidado
            
        Returns:
            DataFrame de pandas con resultados por pregunta
        """
        evaluation_data = self.load_evaluation_data(consolidated_file)
        if not evaluation_data or 'question_by_question' not in evaluation_data:
            return pd.DataFrame()
        
        questions = evaluation_data['question_by_question']
        all_question_data = []
        
        for question_data in questions:
            question_id = question_data.get('question_id', 0)
            question = question_data.get('question', '')
            ground_truth = question_data.get('ground_truth', '')
            rag_results = question_data.get('rag_results', {})
            
            for rag_type, rag_result in rag_results.items():
                if rag_result:
                    row_data = {
                        'question_id': question_id,
                        'question': question,
                        'ground_truth': ground_truth,
                        'rag_type': rag_type,
                        'rag_name': self.rag_names.get(rag_type, rag_type),
                        'answer': rag_result.get('answer', ''),
                        'contexts_count': rag_result.get('contexts_count', 0),
                        'faithfulness': round(rag_result.get('metrics', {}).get('faithfulness', 0), 4),
                        'answer_relevancy': round(rag_result.get('metrics', {}).get('answer_relevancy', 0), 4),
                        'context_precision': round(rag_result.get('metrics', {}).get('context_precision', 0), 4),
                        'context_recall': round(rag_result.get('metrics', {}).get('context_recall', 0), 4),
                        'average_score': round(rag_result.get('metrics', {}).get('average_score', 0), 4),
                        'input_tokens': rag_result.get('input_tokens', 0),
                        'output_tokens': rag_result.get('output_tokens', 0),
                        'total_tokens': rag_result.get('input_tokens', 0) + rag_result.get('output_tokens', 0),
                        'cost': round(rag_result.get('cost', 0), 6),
                        'execution_time': round(rag_result.get('execution_time', 0), 3)
                    }
                    all_question_data.append(row_data)
        
        if not all_question_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_question_data)
        return df.sort_values(['question_id', 'average_score'], ascending=[True, False]).reset_index(drop=True)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str = None) -> Path:
        """
        Guardar el DataFrame en un archivo CSV.
        
        Args:
            df (pd.DataFrame): DataFrame a guardar
            filename (str): Nombre del archivo (opcional)
            
        Returns:
            Path del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_comparison_{timestamp}.csv"
        
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return output_path


def get_rag_comparison_dataframe(results_dir: str = "results") -> pd.DataFrame:
    """
    Función de conveniencia para obtener el DataFrame comparativo de RAGs.
    
    Args:
        results_dir (str): Directorio donde están los JSON de evaluación
        
    Returns:
        DataFrame de pandas con la comparación de RAGs
    """
    comparator = RAGComparisonTable(results_dir)
    consolidated_file = comparator.find_latest_consolidated_file()
    
    if not consolidated_file:
        print("❌ No se encontró archivo consolidado de evaluación")
        return pd.DataFrame()
    
    return comparator.generate_comparison_dataframe(consolidated_file)


def get_question_by_question_dataframe(results_dir: str = "results") -> pd.DataFrame:
    """
    Función de conveniencia para obtener el DataFrame detallado por pregunta.
    
    Args:
        results_dir (str): Directorio donde están los JSON de evaluación
        
    Returns:
        DataFrame de pandas con resultados por pregunta
    """
    comparator = RAGComparisonTable(results_dir)
    consolidated_file = comparator.find_latest_consolidated_file()
    
    if not consolidated_file:
        print("❌ No se encontró archivo consolidado de evaluación")
        return pd.DataFrame()
    
    return comparator.generate_question_by_question_dataframe(consolidated_file)


def main():
    """Función principal - Genera DataFrames comparativos de RAGs desde archivo consolidado"""
    # Crear generador de comparaciones
    comparator = RAGComparisonTable()
    
    print("🔍 Buscando archivo consolidado de evaluación...")
    
    # Buscar archivo consolidado más reciente
    consolidated_file = comparator.find_latest_consolidated_file()
    
    if not consolidated_file:
        print("❌ No se encontró archivo consolidado de evaluación")
        return
    
    print(f"\n📊 Generando DataFrames comparativos...")
    
    # Generar DataFrame comparativo general
    comparison_df = comparator.generate_comparison_dataframe(consolidated_file)
    
    if comparison_df.empty:
        print("❌ No se pudieron cargar datos de evaluación")
        return
    
    # Mostrar el DataFrame general
    print("\n" + "="*120)
    print("📊 COMPARACIÓN GENERAL DE SISTEMAS RAG")
    print("="*120)
    print(comparison_df.to_string(index=False))
    
    # Guardar DataFrame general
    output_path = comparator.save_dataframe(comparison_df, "rag_comparison_general.csv")
    print(f"\n💾 DataFrame general guardado en: {output_path}")
    
    # Generar DataFrame detallado por pregunta
    question_df = comparator.generate_question_by_question_dataframe(consolidated_file)
    
    if not question_df.empty:
        print("\n" + "="*120)
        print("📋 RESULTADOS DETALLADOS POR PREGUNTA")
        print("="*120)
        print(question_df.to_string(index=False, max_colwidth=50))
        
        # Guardar DataFrame detallado
        question_output_path = comparator.save_dataframe(question_df, "rag_comparison_questions.csv")
        print(f"\n💾 DataFrame detallado guardado en: {question_output_path}")
    
    return comparison_df, question_df


if __name__ == "__main__":
    main()
