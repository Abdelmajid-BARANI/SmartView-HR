import spacy
import re
import fitz  # PyMuPDF

class CustomResumeParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self.extract_text()
        self.nlp = spacy.load('en_core_web_sm')
        self.doc = self.nlp(self.text)

    def extract_text(self):
        """Extract text from PDF using PyMuPDF."""
        text = ""
        with fitz.open(self.pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def extract_name(self):
        """Extract candidate's name (first PERSON entity)."""
        for ent in self.doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    def extract_email(self):
        """Extract email using regex."""
        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', self.text)
        return match.group(0) if match else None

    def extract_phone(self):
        """Extract phone number using regex."""
        match = re.search(r'(\+?\d{1,4}[\s-]?)?(\(?\d{3}\)?[\s-]?)?[\d\s-]{7,15}', self.text)
        return match.group(0) if match else None

    def extract_skills(self):
        """Basic skills matching."""
        skills_list = [
            'python', 'java', 'c++', 'c#', 'sql', 'html', 'css', 'javascript',
            'machine learning', 'deep learning', 'django', 'flask', 'pytorch', 'tensorflow',
            'react', 'angular', 'node.js', 'docker', 'kubernetes', 'aws', 'azure', 'git',
            'linux', 'data analysis', 'data science', 'power bi', 'excel', 'tableau'
        ]
        skills_found = []
        for token in self.doc:
            if token.text.lower() in skills_list and token.text.lower() not in skills_found:
                skills_found.append(token.text.lower())
        return skills_found

    def get_extracted_data(self):
        """Return all extracted fields."""
        return {
            "name": self.extract_name(),
            "email": self.extract_email(),
            "mobile_number": self.extract_phone(),
            "skills": self.extract_skills(),
            "no_of_pages": self.count_pages()
        }
    
    def count_pages(self):
        """Count pages using PyMuPDF."""
        with fitz.open(self.pdf_path) as doc:
            return len(doc)
