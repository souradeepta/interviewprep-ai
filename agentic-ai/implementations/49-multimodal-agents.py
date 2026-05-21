"""
Auto-generated from 49-multimodal-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Multimodal Agents
# ## Learning Objectives
# 1. Process and fuse multiple data modalities (text, images, audio)
# 2. Implement cross-modal understanding and alignment
# ======================================================================

# ======================================================================
# ## Level 1: Basic Multimodal Fusion
# ======================================================================

import numpy as np
from typing import Dict, Tuple, List

class BasicMultimodalAgent:
    """Simple multimodal agent with feature extraction and fusion."""
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract simple text features (word count, length, etc)."""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'char_density': len(text) / max(len(text.split()), 1),
        }
        return np.array(list(features.values()))
    
    def extract_image_features(self, image_array: np.ndarray) -> np.ndarray:
        """Extract simple image features (shape, brightness, contrast)."""
        if len(image_array.shape) == 3:
            brightness = np.mean(image_array)
            contrast = np.std(image_array)
            height, width = image_array.shape[:2]
        else:
            brightness = np.mean(image_array)
            contrast = np.std(image_array)
            height, width = image_array.shape
        
        features = {
            'brightness': brightness,
            'contrast': contrast,
            'aspect_ratio': height / max(width, 1),
        }
        return np.array(list(features.values()))
    
    def fuse_features(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """Simple concatenation fusion."""
        # Normalize to same scale
        text_norm = (text_features - np.mean(text_features)) / (np.std(text_features) + 1e-6)
        image_norm = (image_features - np.mean(image_features)) / (np.std(image_features) + 1e-6)
        return np.concatenate([text_norm, image_norm])
    
    def analyze(self, image: np.ndarray, text: str) -> Dict:
        """Analyze image and text together."""
        text_feats = self.extract_text_features(text)
        image_feats = self.extract_image_features(image)
        fused = self.fuse_features(text_feats, image_feats)
        
        return {
            'text_features': text_feats,
            'image_features': image_feats,
            'fused_vector': fused,
            'fusion_norm': np.linalg.norm(fused)
        }

# Test
agent = BasicMultimodalAgent()
test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
test_text = "This is a photo of a cat sitting on a mat."
result = agent.analyze(test_image, test_text)
print(f"Fused vector shape: {result['fused_vector'].shape}")
print(f"Fusion norm: {result['fusion_norm']:.2f}")


# ======================================================================
# ## Level 2: Advanced Multimodal with Alignment and Confidence
# ======================================================================

import json
from dataclasses import dataclass
from enum import Enum

class Modality(Enum):
    TEXT = 'text'
    IMAGE = 'image'
    AUDIO = 'audio'
    VIDEO = 'video'

@dataclass
class ModalityOutput:
    modality: Modality
    features: np.ndarray
    confidence: float
    processing_time: float

class AdvancedMultimodalAgent:
    """Advanced agent with confidence scores and fallback handling."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.modality_weights = {Modality.TEXT: 0.3, Modality.IMAGE: 0.5, Modality.AUDIO: 0.2}
    
    def process_text(self, text: str) -> ModalityOutput:
        """Process text with confidence estimation."""
        features = np.array([len(text), len(text.split()), len(set(text.split()))])
        # Confidence based on text length and vocabulary diversity
        vocab_ratio = len(set(text.split())) / max(len(text.split()), 1)
        confidence = min(1.0, len(text) / 100 * vocab_ratio)
        return ModalityOutput(
            modality=Modality.TEXT,
            features=features,
            confidence=confidence,
            processing_time=0.01
        )
    
    def process_image(self, image: np.ndarray) -> ModalityOutput:
        """Process image with confidence estimation."""
        if image.size == 0:
            return ModalityOutput(Modality.IMAGE, np.zeros(3), 0.0, 0.0)
        
        features = np.array([
            np.mean(image),
            np.std(image),
            np.max(image) - np.min(image)
        ])
        # Confidence based on image complexity (contrast)
        contrast = np.std(image)
        confidence = min(1.0, contrast / 100)
        return ModalityOutput(
            modality=Modality.IMAGE,
            features=features,
            confidence=confidence,
            processing_time=0.05
        )
    
    def align_modalities(self, outputs: List[ModalityOutput]) -> Dict:
        """Align and weight modalities by confidence."""
        total_confidence = sum(o.confidence for o in outputs)
        if total_confidence == 0:
            weights = {o.modality: 1/len(outputs) for o in outputs}
        else:
            weights = {o.modality: o.confidence / total_confidence for o in outputs}
        
        alignment = {}
        for output in outputs:
            if output.confidence >= self.confidence_threshold:
                alignment[output.modality.value] = {
                    'features': output.features.tolist(),
                    'confidence': output.confidence,
                    'weight': weights[output.modality],
                    'reliable': True
                }
            else:
                alignment[output.modality.value] = {
                    'features': output.features.tolist(),
                    'confidence': output.confidence,
                    'weight': weights[output.modality],
                    'reliable': False
                }
        return alignment
    
    def fuse_with_weights(self, outputs: List[ModalityOutput]) -> np.ndarray:
        """Weighted fusion based on modality confidence."""
        alignment = self.align_modalities(outputs)
        
        fused = np.zeros(max(len(o.features) for o in outputs))
        total_weight = 0
        
        for output in outputs:
            weight = alignment[output.modality.value]['weight']
            padded_features = np.zeros(len(fused))
            padded_features[:len(output.features)] = output.features
            fused += weight * padded_features
            total_weight += weight
        
        return fused / max(total_weight, 1e-6)
    
    def analyze(self, image: np.ndarray, text: str) -> Dict:
        """End-to-end multimodal analysis."""
        text_output = self.process_text(text)
        image_output = self.process_image(image)
        
        outputs = [text_output, image_output]
        alignment = self.align_modalities(outputs)
        fused = self.fuse_with_weights(outputs)
        
        return {
            'alignment': alignment,
            'fused_vector': fused.tolist(),
            'decision': 'reliable' if all(v['reliable'] for v in alignment.values()) else 'uncertain'
        }

# Test
adv_agent = AdvancedMultimodalAgent()
test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
test_text = "A detailed description of an image."
result = adv_agent.analyze(test_image, test_text)
print(f"Decision: {result['decision']}")
print(f"Alignment keys: {list(result['alignment'].keys())}")


# ======================================================================
# ## Real-World Example 1: Visual Question Answering
# ======================================================================

class VisualQAAgent:
    """Visual question answering agent."""
    
    def __init__(self):
        self.question_types = {
            'what': 'object_detection',
            'where': 'spatial_reasoning',
            'how many': 'counting',
            'why': 'semantic_reasoning'
        }
    
    def classify_question(self, question: str) -> str:
        """Classify question type."""
        q_lower = question.lower()
        for keyword, qtype in self.question_types.items():
            if keyword in q_lower:
                return qtype
        return 'general'
    
    def extract_image_regions(self, image: np.ndarray) -> Dict:
        """Extract regions of interest from image."""
        # Simplified: divide into grid regions
        h, w = image.shape[:2]
        regions = {}
        for i in range(2):
            for j in range(2):
                y_start, y_end = int(i * h/2), int((i+1) * h/2)
                x_start, x_end = int(j * w/2), int((j+1) * w/2)
                region_img = image[y_start:y_end, x_start:x_end]
                regions[f'region_{i}_{j}'] = {
                    'brightness': np.mean(region_img),
                    'area': region_img.size
                }
        return regions
    
    def answer_question(self, image: np.ndarray, question: str) -> Dict:
        """Answer visual question."""
        q_type = self.classify_question(question)
        regions = self.extract_image_regions(image)
        
        if q_type == 'counting':
            count = sum(1 for r in regions.values() if r['brightness'] > 128)
            answer = f"{count} bright regions detected"
        elif q_type == 'spatial_reasoning':
            brightest = max(regions.items(), key=lambda x: x[1]['brightness'])
            answer = f"Brightest region at {brightest[0]}"
        else:
            answer = "General description of image content"
        
        return {
            'question': question,
            'question_type': q_type,
            'answer': answer,
            'confidence': 0.8 if q_type != 'general' else 0.5
        }

# Test
vqa = VisualQAAgent()
test_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
result1 = vqa.answer_question(test_img, "What objects are in the image?")
result2 = vqa.answer_question(test_img, "Where is the brightest region?")
print(f"Q1 Type: {result1['question_type']}, Answer: {result1['answer']}")
print(f"Q2 Type: {result2['question_type']}, Answer: {result2['answer']}")


# ======================================================================
# ## Real-World Example 2: Medical Image Analysis
# ======================================================================

class MedicalImageAgent:
    """Multimodal medical analysis (images + clinical notes)."""
    
    def __init__(self):
        self.risk_levels = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    
    def analyze_image(self, medical_image: np.ndarray) -> Dict:
        """Extract features from medical image."""
        # Simplified: look for abnormalities (high variance regions)
        h, w = medical_image.shape[:2]
        # Check center region (common abnormality location)
        center = medical_image[h//4:3*h//4, w//4:3*w//4]
        abnormality_score = np.std(center) / 255  # Normalize
        
        return {
            'abnormality_score': abnormality_score,
            'region_clarity': 1 - (np.std(medical_image) / 255),
            'recommendation': 'further_review' if abnormality_score > 0.3 else 'normal'
        }
    
    def analyze_notes(self, clinical_notes: str) -> Dict:
        """Extract risk factors from clinical notes."""
        risk_keywords = {
            'tumor': 3,
            'fracture': 2,
            'infection': 2,
            'inflammation': 1,
            'normal': 0
        }
        
        max_risk = 0
        found_keywords = []
        for keyword, risk in risk_keywords.items():
            if keyword in clinical_notes.lower():
                max_risk = max(max_risk, risk)
                found_keywords.append(keyword)
        
        return {
            'clinical_risk': max_risk,
            'keywords': found_keywords,
            'confidence': len(found_keywords) / 3
        }
    
    def multimodal_diagnosis(self, image: np.ndarray, notes: str) -> Dict:
        """Combine image and text for diagnosis."""
        image_analysis = self.analyze_image(image)
        text_analysis = self.analyze_notes(notes)
        
        # Combined risk score
        image_risk = image_analysis['abnormality_score'] * 3
        text_risk = text_analysis['clinical_risk']
        combined_risk = (image_risk + text_risk) / 2
        
        # Determine severity
        if combined_risk >= 2:
            severity = 'severe'
        elif combined_risk >= 1.5:
            severity = 'moderate'
        elif combined_risk >= 0.5:
            severity = 'mild'
        else:
            severity = 'normal'
        
        return {
            'image_analysis': image_analysis,
            'text_analysis': text_analysis,
            'combined_risk': combined_risk,
            'severity': severity,
            'recommendation': 'urgent_review' if severity == 'severe' else 'standard_review'
        }

# Test
medical_agent = MedicalImageAgent()
test_medical_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
test_notes = "CT scan shows abnormal findings. Possible tumor detected in center region."
diagnosis = medical_agent.multimodal_diagnosis(test_medical_img, test_notes)
print(f"Severity: {diagnosis['severity']}")
print(f"Recommendation: {diagnosis['recommendation']}")
print(f"Combined Risk: {diagnosis['combined_risk']:.2f}")


# ======================================================================
# ## Real-World Example 3: Content Moderation with Multimodal Context
# ======================================================================

class MultimodalModerationAgent:
    """Content moderation considering both image and text."""
    
    def __init__(self):
        self.text_toxicity_keywords = {
            'hate': 1.0,
            'violence': 0.9,
            'harassment': 0.8,
            'abuse': 0.7
        }
        self.moderation_threshold = 0.6
    
    def assess_text_toxicity(self, text: str) -> float:
        """Assess toxicity of text."""
        max_toxicity = 0
        text_lower = text.lower()
        for keyword, score in self.text_toxicity_keywords.items():
            if keyword in text_lower:
                max_toxicity = max(max_toxicity, score)
        return max_toxicity
    
    def assess_image_risk(self, image: np.ndarray) -> float:
        """Assess risk level of image (simplified)."""
        # High variance = potentially graphic/disturbing
        variance = np.std(image) / 255
        # Dark images more likely problematic
        darkness = 1 - (np.mean(image) / 255)
        image_risk = (variance + darkness) / 2
        return image_risk
    
    def context_matters(self, text: str) -> bool:
        """Check if text provides educational/news context."""
        educational_keywords = ['educational', 'news', 'documentary', 'awareness', 'historical']
        return any(keyword in text.lower() for keyword in educational_keywords)
    
    def moderate(self, image: np.ndarray, caption: str) -> Dict:
        """Multimodal content moderation decision."""
        text_toxicity = self.assess_text_toxicity(caption)
        image_risk = self.assess_image_risk(image)
        has_context = self.context_matters(caption)
        
        # Combined score
        combined_score = (text_toxicity + image_risk) / 2
        
        # Reduce score if educational context
        if has_context:
            combined_score *= 0.7
        
        # Decision
        if combined_score >= self.moderation_threshold:
            decision = 'FLAG'
            action = 'remove'
        else:
            decision = 'ALLOW'
            action = 'publish'
        
        return {
            'text_toxicity': text_toxicity,
            'image_risk': image_risk,
            'has_context': has_context,
            'combined_score': combined_score,
            'decision': decision,
            'action': action,
            'confidence': min(1.0, combined_score + 0.2)
        }

# Test
mod_agent = MultimodalModerationAgent()
test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
result1 = mod_agent.moderate(test_img, "Check out this cool educational documentary on history.")
result2 = mod_agent.moderate(test_img, "This is hate speech and violence.")
print(f"Post 1 - Decision: {result1['decision']}, Action: {result1['action']}")
print(f"Post 2 - Decision: {result2['decision']}, Action: {result2['action']}")


# ======================================================================
# ## Key Takeaways
# 1. **Modality Fusion**: Combine features from different modalities through concatenation or weighted fusion
# 2. **Confidence Tracking**: Each modality has its own confidence; use alignment to determine reliability
# 3. **Fallback Strategy**: If one modality fails, degrade gracefully using other modalities
# ======================================================================
