"""
Query Preprocessor: Mejora queries antes del retrieval.

Técnicas implementadas:
1. Query expansion: Añade sinónimos y términos relacionados
2. Query rewriting: Reformula para maximizar recall
3. Spelling correction: Corrige errores ortográficos
4. Stopword removal: Opcional para búsquedas keyword-based
"""

from typing import Optional, Dict, List
import re


class QueryPreprocessor:
    """
    Preprocesa queries para mejorar retrieval.
    
    Métodos:
    - expand: Añade sinónimos/términos relacionados
    - rewrite: Reformula query completamente
    - correct_spelling: Corrige errores comunes
    - preprocess: Pipeline completo
    """
    
    def __init__(self, use_llm: bool = False, llm=None):
        """
        Args:
            use_llm: Si True, usa LLM para rewriting (requiere llm)
            llm: Instancia de LLM (ChatOpenAI, Claude, etc.)
        """
        self.use_llm = use_llm
        self.llm = llm
        
        # Diccionario de sinónimos (específico del dominio)
        self.synonyms = {
            "benzoato": ["benzoato de sodio", "ácido benzoico", "E211"],
            "sorbato": ["sorbato de potasio", "ácido sórbico", "E202"],
            "preservante": ["conservante", "antimicrobiano", "agente conservador"],
            "natural": ["orgánico", "bio", "de origen vegetal"],
            "sintético": ["artificial", "químico"],
            "levadura": ["levaduras", "yeast", "Saccharomyces"],
            "moho": ["hongos", "mohos", "mold", "Aspergillus"],
            "bacteria": ["bacterias", "bacteria", "microorganismo bacteriano"],
            "limite": ["límite", "concentración máxima", "dosis máxima"],
            "pH": ["acidez", "nivel de pH", "condición de pH"],
            "aW": ["actividad de agua", "aw", "water activity"],
        }
        
        # Correcciones ortográficas comunes
        self.spelling_corrections = {
            "bensoato": "benzoato",
            "sorbate": "sorbato",
            "preservativo": "preservante",
            "conservador": "conservante",
            "acides": "acidez",
            "limete": "límite",
        }
    
    def expand(self, query: str) -> str:
        """
        Expande query añadiendo sinónimos.
        
        Ejemplo:
            Input: "benzoato en pH 4.5"
            Output: "benzoato benzoato de sodio ácido benzoico E211 en pH 4.5"
        """
        expanded_terms = []
        words = query.lower().split()
        
        for word in words:
            # Añadir palabra original
            expanded_terms.append(word)
            
            # Añadir sinónimos si existen
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word])
        
        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return " ".join(unique_terms)
    
    def rewrite_with_llm(self, query: str) -> str:
        """
        Reescribe query usando LLM para maximizar recall.
        
        Ejemplo:
            Input: "Alternativa a benzoato"
            Output: "Conservante natural alternativo al benzoato de sodio o ácido benzoico"
        """
        if not self.use_llm or self.llm is None:
            return query
        
        prompt = f"""
Eres un experto en conservantes alimentarios. Reescribe esta query de búsqueda para 
maximizar la recuperación de información relevante en una base de datos científica.

REGLAS:
1. Expande abreviaciones técnicas
2. Añade términos técnicos relacionados
3. Mantén números y condiciones exactas (pH, concentraciones)
4. NO cambies el significado original
5. Retorna SOLO la query reescrita, sin explicación

Query original: {query}

Query reescrita:"""
        
        try:
            rewritten = self.llm.invoke(prompt).strip()
            # Limpiar comillas o markdown si el LLM las añade
            rewritten = rewritten.strip('"\'`')
            return rewritten
        except Exception as e:
            print(f"Warning: Error en LLM rewriting: {e}")
            return query
    
    def rewrite_heuristic(self, query: str) -> str:
        """
        Reescritura basada en heurísticas (sin LLM).
        
        Estrategia:
        1. Detecta patrones comunes
        2. Reformula para maximizar recall
        """
        query_lower = query.lower()
        
        # Patrón: "alternativa a X"
        match = re.search(r'alternativa\s+(?:a|al)\s+(\w+)', query_lower)
        if match:
            preservative = match.group(1)
            # Expandir con sinónimos
            synonyms_str = " ".join(self.synonyms.get(preservative, [preservative]))
            return f"conservante natural alternativo {synonyms_str} {query}"
        
        # Patrón: "limite de X"
        match = re.search(r'l[ií]mite\s+(?:de|del)\s+(\w+)', query_lower)
        if match:
            preservative = match.group(1)
            return f"concentración máxima dosis límite {preservative} {query}"
        
        # Patrón: "X contra Y"
        match = re.search(r'(\w+)\s+contra\s+(\w+)', query_lower)
        if match:
            preservative = match.group(1)
            microorg = match.group(2)
            return f"eficacia antimicrobiana {preservative} inhibición {microorg} {query}"
        
        # Si no coincide ningún patrón, retornar original
        return query
    
    def correct_spelling(self, query: str) -> str:
        """
        Corrige errores ortográficos comunes.
        
        Ejemplo:
            Input: "bensoato en acides 4.5"
            Output: "benzoato en acidez 4.5"
        """
        words = query.split()
        corrected = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.spelling_corrections:
                # Mantener capitalización original
                corrected_word = self.spelling_corrections[word_lower]
                if word[0].isupper():
                    corrected_word = corrected_word.capitalize()
                corrected.append(corrected_word)
            else:
                corrected.append(word)
        
        return " ".join(corrected)
    
    def extract_conditions(self, query: str) -> Dict[str, any]:
        """
        Extrae condiciones estructuradas de la query.
        
        Útil para búsqueda híbrida con filtros.
        
        Returns:
            Dict con pH, aW, microorganismos detectados
        """
        conditions = {
            "pH": None,
            "aW": None,
            "microorganisms": [],
            "preservatives": []
        }
        
        # Extraer pH
        ph_match = re.search(r'pH\s*[:\s]*(\d+\.?\d*)', query, re.IGNORECASE)
        if ph_match:
            conditions["pH"] = float(ph_match.group(1))
        
        # Extraer aW
        aw_match = re.search(r'(?:aW|aw|actividad de agua)\s*[:\s]*(\d+\.?\d*)', query, re.IGNORECASE)
        if aw_match:
            conditions["aW"] = float(aw_match.group(1))
        
        # Detectar microorganismos
        microorg_keywords = ["levadura", "moho", "bacteria", "Saccharomyces", "Aspergillus", "Zygosaccharomyces"]
        for keyword in microorg_keywords:
            if keyword.lower() in query.lower():
                conditions["microorganisms"].append(keyword)
        
        # Detectar preservantes
        preserv_keywords = ["benzoato", "sorbato", "nisina", "clavo", "orégano", "romero"]
        for keyword in preserv_keywords:
            if keyword.lower() in query.lower():
                conditions["preservatives"].append(keyword)
        
        return conditions
    
    def preprocess(
        self, 
        query: str,
        expand: bool = True,
        rewrite: bool = True,
        correct_spelling: bool = True
    ) -> str:
        """
        Pipeline completo de preprocesamiento.
        
        Args:
            query: Query original
            expand: Si True, expande con sinónimos
            rewrite: Si True, reescribe query
            correct_spelling: Si True, corrige ortografía
        
        Returns:
            Query procesada
        """
        processed = query
        
        # 1. Corrección ortográfica
        if correct_spelling:
            processed = self.correct_spelling(processed)
        
        # 2. Reescritura
        if rewrite:
            if self.use_llm:
                processed = self.rewrite_with_llm(processed)
            else:
                processed = self.rewrite_heuristic(processed)
        
        # 3. Expansión (opcional, puede hacer queries muy largas)
        if expand:
            processed = self.expand(processed)
        
        return processed


# Ejemplo de uso
if __name__ == "__main__":
    # Sin LLM (heurísticas)
    preprocessor = QueryPreprocessor(use_llm=False)
    
    queries = [
        "Alternativa a benzoato en pH 4.5",
        "Limite de sorbato contra levaduras",
        "bensoato para productos con acides baja",
    ]
    
    for query in queries:
        print(f"\nOriginal: {query}")
        print(f"Corregida: {preprocessor.correct_spelling(query)}")
        print(f"Reescrita: {preprocessor.rewrite_heuristic(query)}")
        print(f"Expandida: {preprocessor.expand(query)}")
        print(f"Condiciones: {preprocessor.extract_conditions(query)}")
        print(f"Pipeline completo: {preprocessor.preprocess(query)}")
