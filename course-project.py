# course project IR

# Gtoup Members

# 1- Momna Waryam khan - MSCS25011
# 2- Talal Ahmad – MSCS25015


# Domain-Specific IR System for Legal Document Retrieval with
# Intelligent Query Expansion



# it directlys loads the data from kaggle

import pandas as pd
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import warnings
import os
import json
import re

warnings.filterwarnings('ignore')

# Download required NLTK data (silently)
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class PakistanLegalIR:
    def __init__(self):
        self.documents = []
        self.doc_metadata = []
        self.vectorizer = None
        self.doc_vectors = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.legal_stop_words = {'shall', 'may', 'must', 'said', 'thereof', 
                                  'herein', 'hereby', 'hereof', 'thereto',
                                  'aforesaid', 'aforementioned', 'wherein'}
        self.stop_words.update(self.legal_stop_words)
        
    def is_table_of_contents(self, text):
        """Check if text is a table of contents"""
        text_lower = text.lower()
        
        # Check for TOC patterns
        toc_patterns = [
            r'^\s*contents\s*$',
            r'^\s*table\s+of\s+contents\s*$',
            r'chapter\s+[ivxlcdm]+',
            r'preliminary',
        ]
        
        for pattern in toc_patterns:
            if re.search(pattern, text_lower, re.MULTILINE):
                return True
        
        # Check if mostly numbered lists
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < 5:
            return False
            
        numbered_lines = sum(1 for line in lines if re.match(r'^\d+\.', line))
        
        if numbered_lines / len(lines) > 0.5:
            return True
        
        # Check for common TOC keywords
        toc_keywords = ['short title', 'definitions', 'extent and commencement', 
                       'punishment', 'cognizance', 'application of', 'power to make',
                       'sale and repair', 'unlicensed', 'power to prohibit']
        keyword_count = sum(1 for keyword in toc_keywords if keyword in text_lower)
        
        # Stricter TOC detection
        if keyword_count >= 3 and len(text.split()) < 500:
            if 'chapter' in text_lower or 'preliminary' in text_lower:
                return True
        
        return False
        
    def download_and_load_dataset(self):
        """Download dataset using kagglehub and load it"""
        try:
            print("Loading dataset from Kaggle...", end=" ", flush=True)
            
            import kagglehub
            dataset_path = kagglehub.dataset_download("ayeshajadoon/pakistan-law-data")
            
            all_files = []
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    all_files.append(os.path.join(root, file))
            
            if not all_files:
                print("ERROR: No files found")
                return False
            
            # Try all files, not just the first one
            loaded = False
            for file_path in all_files:
                ext = os.path.splitext(file_path)[1].lower()
                
                try:
                    if ext == '.parquet':
                        df = pd.read_parquet(file_path)
                        if self._process_dataframe(df):
                            loaded = True
                    elif ext == '.csv':
                        df = pd.read_csv(file_path)
                        if self._process_dataframe(df):
                            loaded = True
                    elif ext == '.json' or ext == '.jsonl':
                        if self._load_json_file(file_path):
                            loaded = True
                    elif ext in ['.txt']:
                        if self._load_text_file(file_path):
                            loaded = True
                except Exception as e:
                    continue
            
            if not loaded:
                print("ERROR: Failed to load")
            
            return loaded
            
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    def _load_json_file(self, file_path):
        """Load JSON/JSONL file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except:
                    f.seek(0)
                    data = [json.loads(line) for line in f if line.strip()]
            
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    text_fields = ['content', 'text', 'document', 'body', 'description']
                    text = None
                    
                    for field in text_fields:
                        if field in item:
                            text = str(item[field])
                            break
                    
                    if not text:
                        for key, val in item.items():
                            if isinstance(val, str) and len(val) > 100:
                                text = val
                                break
                    
                    if text and len(text.strip()) > 50 and not self.is_table_of_contents(text):
                        self.documents.append(text)
                        
                        meta = {'doc_id': idx}
                        meta_fields = ['title', 'filename', 'file_name', 'name', 
                                     'category', 'law_name', 'act_name']
                        for field in meta_fields:
                            if field in item:
                                meta[field] = str(item[field])
                        self.doc_metadata.append(meta)
            
            return len(self.documents) > 0
            
        except Exception:
            return False
    
    def _load_text_file(self, file_path):
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            docs = [doc.strip() for doc in content.split('\n\n') 
                   if len(doc.strip()) > 100 and not self.is_table_of_contents(doc)]
            
            if not docs:
                docs = [doc.strip() for doc in content.split('\n') 
                       if len(doc.strip()) > 100 and not self.is_table_of_contents(doc)]
            
            self.documents = docs
            self.doc_metadata = [{'doc_id': i, 'source': os.path.basename(file_path)} 
                               for i in range(len(docs))]
            
            return len(self.documents) > 0
            
        except Exception:
            return False
    
    def _process_dataframe(self, df):
        """Process pandas DataFrame - append to existing documents"""
        text_col_names = ['text', 'content', 'document', 'body', 'description', 
                        'law_text', 'legal_text', 'passage', 'article']
        
        text_col = None
        for col in text_col_names:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            string_cols = df.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                text_col = string_cols[0]
        
        if text_col is None:
            return False
        
        # Store starting index for metadata
        start_idx = len(self.documents)
        
        metadata_cols = ['title', 'category', 'law_name', 'act_name', 'section', 
                        'filename', 'file_name']
        
        # Append documents instead of replacing
        for idx, row in df.iterrows():
            doc_text = str(row[text_col])
            
            if len(doc_text.strip()) > 50 and not self.is_table_of_contents(doc_text):
                self.documents.append(doc_text)
                
                meta = {'doc_id': start_idx + len(self.documents) - 1}
                for col in metadata_cols:
                    if col in df.columns:
                        meta[col] = str(row[col])
                self.doc_metadata.append(meta)
        
        return len(self.documents) > 0
    
    def preprocess_text(self, text):
        """Advanced preprocessing for legal text"""
        text = str(text).lower()
        text = ''.join([char if char not in string.punctuation else ' ' 
                       for char in text])
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and len(word) > 2 and word.isalpha()]
        return " ".join(words)
    
    def expand_query_wordnet(self, query):
        """Expand query using WordNet synonyms"""
        words = query.lower().split()
        expanded_terms = set(words)
        
        for word in words:
            synsets = wordnet.synsets(word)
            for syn in synsets[:2]:
                for lemma in syn.lemmas()[:3]:
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() not in self.stop_words:
                        expanded_terms.add(synonym.lower())
        
        return " ".join(expanded_terms)
    
    def expand_query_legal_terms(self, query):
        """Expand query with Pakistan legal domain terms"""
        legal_expansions = {
            'contract': ['agreement', 'covenant', 'obligation', 'undertaking', 'deed'],
            'breach': ['violation', 'infringement', 'default', 'non-compliance'],
            'damages': ['compensation', 'remedy', 'restitution', 'indemnity'],
            'crime': ['offense', 'offence', 'violation', 'wrongdoing'],
            'punishment': ['penalty', 'sanction', 'sentence', 'fine'],
            'accused': ['defendant', 'respondent', 'suspect'],
            'plaintiff': ['claimant', 'petitioner', 'complainant'],
            'liability': ['responsibility', 'accountability', 'obligation'],
            'court': ['tribunal', 'judiciary', 'forum', 'bench'],
            'law': ['statute', 'regulation', 'ordinance', 'legislation', 'act'],
            'rights': ['entitlement', 'privilege', 'claim', 'freedom'],
            'evidence': ['proof', 'testimony', 'documentation'],
        }
        
        expanded_terms = set(query.lower().split())
        for word in query.lower().split():
            if word in legal_expansions:
                expanded_terms.update(legal_expansions[word])
        
        return " ".join(expanded_terms)
    
    def build_index(self):
        """Build TF-IDF index"""
        print("Building TF-IDF index...", end=" ", flush=True)
        
        preprocessed_docs = [self.preprocess_text(doc) for doc in self.documents]
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        
        self.doc_vectors = self.vectorizer.fit_transform(preprocessed_docs)
        print("Done")
    
    def extract_page_and_line_info(self, doc_text):
        """Extract page number and line information from document"""
        lines = doc_text.split('\n')
        
        # Try to find page number
        page_num = None
        for i, line in enumerate(lines[:10]):
            page_match = re.search(r'page\s+(\d+)\s+of\s+(\d+)', line, re.IGNORECASE)
            if page_match:
                page_num = f"Page {page_match.group(1)} of {page_match.group(2)}"
                break
        
        # Count total lines
        total_lines = len([l for l in lines if l.strip()])
        
        return page_num, total_lines
    
    def get_clean_preview(self, doc_text, max_chars=300):
        """Get preview without table of contents"""
        lines = doc_text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip TOC-like lines
            if re.match(r'^\d+\.\s+[A-Z]', line):
                continue
            if re.match(r'^chapter\s+[ivxlcdm]+', line, re.IGNORECASE):
                continue
            if 'preliminary' in line.lower() and len(line) < 30:
                continue
            if re.match(r'^page\s+\d+\s+of\s+\d+', line, re.IGNORECASE):
                continue
                
            clean_lines.append(line)
            
            if len(' '.join(clean_lines)) >= max_chars:
                break
        
        preview = ' '.join(clean_lines)[:max_chars]
        return preview if preview else doc_text[:max_chars]
    
    def count_meaningful_words(self, query, doc_text):
        """Count meaningful query words in document"""
        query_words = [w.lower() for w in query.split() 
                      if w.lower() not in ['the', 'of', 'and', 'in', 'are', 'a', 'an', 'to', 'for', 'is', 'what']]
        
        if not query_words:
            return {}, 0
        
        doc_lower = doc_text.lower()
        word_counts = {}
        
        for word in query_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            count = len(re.findall(pattern, doc_lower))
            if count > 0:
                word_counts[word] = count
        
        return word_counts, sum(word_counts.values())
    
    def search(self, query, top_k=5, use_expansion=True):
        """Search with query expansion"""
        if use_expansion:
            expanded_wn = self.expand_query_wordnet(query)
            expanded_legal = self.expand_query_legal_terms(query)
            final_query = f"{query} {query} {expanded_legal} {expanded_wn}"
        else:
            final_query = query
        
        preprocessed_query = self.preprocess_text(final_query)
        query_vector = self.vectorizer.transform([preprocessed_query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                word_counts, total = self.count_meaningful_words(query, self.documents[idx])
                page_info, total_lines = self.extract_page_and_line_info(self.documents[idx])
                clean_preview = self.get_clean_preview(self.documents[idx], max_chars=300)
                
                results.append({
                    'rank': len(results) + 1,
                    'doc_id': idx,
                    'score': similarities[idx],
                    'word_counts': word_counts,
                    'total_matches': total,
                    'page_info': page_info,
                    'total_lines': total_lines,
                    'snippet': clean_preview,
                    'metadata': self.doc_metadata[idx]
                })
        
        return results
    
    def display_results(self, query, results):
        """Display results in compact format"""
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
        
        if not results:
            print(f"\nRESULT: No documents found")
            print(f"Total documents searched: {len(self.documents)}")
            print("="*80 + "\n")
            return
        
        print(f"\nRESULT: Found in {len(results)} document(s) out of {len(self.documents)} total")
        
        # Show files
        files = set()
        for r in results:
            fname = r['metadata'].get('filename', r['metadata'].get('file_name', 
                    r['metadata'].get('title', 'Unknown')))
            files.add(fname)
        
        print(f"\nFiles containing matches: {len(files)}")
        for i, fname in enumerate(sorted(files)[:5], 1):
            print(f"  {i}. {fname}")
        if len(files) > 5:
            print(f"  ... and {len(files)-5} more files")
        
        print("\n" + "="*80)
        
        for r in results:
            fname = r['metadata'].get('filename', r['metadata'].get('file_name', 
                    r['metadata'].get('title', 'Unknown')))
            
            print(f"\n*** RANK #{r['rank']} ***")
            print(f"TF-IDF Score: {r['score']:.4f} ({r['score']*100:.1f}% relevance)")
            print(f"Word Frequency: {r['total_matches']} matches")
            
            if r['word_counts']:
                word_freq_str = ', '.join([f"'{k}':{v}" for k,v in 
                                          sorted(r['word_counts'].items(), 
                                                key=lambda x: x[1], reverse=True)[:5]])
                print(f"  Details: {word_freq_str}")
            
            print(f"Source File: {fname}")
            
            # Show page and line info
            if r['page_info']:
                print(f"Location: {r['page_info']}, Total Lines: {r['total_lines']}")
            else:
                print(f"Total Lines: {r['total_lines']}")
            
            print(f"\nPreview:")
            print(f"{r['snippet']}...")
            print("-"*80)
        
        print("="*80 + "\n")

def main():
    print("\n" + "="*80)
    print("PAKISTAN LEGAL DOCUMENT INFORMATION RETRIEVAL SYSTEM")
    print("="*80 + "\n")
    
    ir_system = PakistanLegalIR()
    
    if not ir_system.download_and_load_dataset():
        print("ERROR: Failed to load dataset")
        return
    
    print(f"SUCCESS: Loaded {len(ir_system.documents)} documents")
    ir_system.build_index()
    
    # Single example query
    print("\n" + "="*80)
    print("EXAMPLE SEARCH")
    print("="*80)
    
    query = "What are the rights of accused in criminal proceedings"
    results = ir_system.search(query, top_k=3)
    ir_system.display_results(query, results)
    
    # Interactive mode
    print("="*80)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        try:
            query = input("Enter your query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting system. Goodbye!")
                break
            if query:
                results = ir_system.search(query, top_k=5)
                ir_system.display_results(query, results)
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting system. Goodbye!")
            break

if __name__ == "__main__":
    main()
    
