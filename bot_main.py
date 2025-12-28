"""
Quality-Aware Buffer of Thoughts - Complete PoC
Addresses: 1) Cold Start Problem, 2) Quality Assurance

Key Innovations:
1. Multi-Model Bootstrap: Use ensemble of models to initialize meta-buffer
2. Verification Test Suite: Validate templates before adding to buffer
3. Quality Scoring System: Track template effectiveness over time
4. Validation Layer: Verify templates before using them
5. Self-Correction: Detect and fix declining templates

Author: Research PoC for BoT Enhancement
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib


# ============================================================================
# PART 1: DATA STRUCTURES
# ============================================================================

class TemplateCategory(Enum):
    MATHEMATICAL = "mathematical"
    CODE_PROGRAMMING = "code_programming"
    TEXT_COMPREHENSION = "text_comprehension"
    COMMON_SENSE = "common_sense"


@dataclass
class TemplateProvenance:
    """Track template origin for quality assurance"""
    source_models: List[str]  # Which models contributed
    creation_date: str
    validation_models: List[str]  # Which models validated
    refinement_history: List[str] = field(default_factory=list)
    parent_template_id: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class QualityMetrics:
    """Comprehensive quality tracking"""
    success_count: int = 0
    failure_count: int = 0
    recent_successes: List[bool] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    verification_passed: int = 0
    verification_total: int = 0
    
    def add_result(self, success: bool, confidence: float = 1.0):
        """Record a usage result"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
        self.recent_successes.append(success)
        self.confidence_scores.append(confidence)
        
        # Keep only last 20 results
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)
            self.confidence_scores.pop(0)
    
    def get_success_rate(self) -> float:
        """Calculate success rate from recent history"""
        if not self.recent_successes:
            return 0.5  # Neutral for new templates
        return sum(self.recent_successes) / len(self.recent_successes)
    
    def get_quality_score(self) -> float:
        """
        Composite quality score combining:
        - Success rate (40%)
        - Verification performance (30%)
        - Confidence (20%)
        - Usage maturity (10%)
        """
        success_rate = self.get_success_rate()
        
        verification_rate = (self.verification_passed / max(self.verification_total, 1))
        
        avg_confidence = (np.mean(self.confidence_scores) 
                         if self.confidence_scores else 0.5)
        
        total_uses = self.success_count + self.failure_count
        maturity_factor = min(total_uses / 50.0, 1.0)  # Max at 50 uses
        
        quality = (0.4 * success_rate + 
                  0.3 * verification_rate + 
                  0.2 * avg_confidence + 
                  0.1 * maturity_factor)
        
        return quality
    
    def get_trend(self) -> str:
        """Detect performance trend"""
        if len(self.recent_successes) < 10:
            return "insufficient_data"
        
        mid = len(self.recent_successes) // 2
        older_rate = sum(self.recent_successes[:mid]) / mid
        newer_rate = sum(self.recent_successes[mid:]) / (len(self.recent_successes) - mid)
        
        if newer_rate > older_rate + 0.15:
            return "improving"
        elif newer_rate < older_rate - 0.15:
            return "declining"
        return "stable"
    
    def to_dict(self):
        return {
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.get_success_rate(),
            'quality_score': self.get_quality_score(),
            'trend': self.get_trend(),
            'verification_rate': self.verification_passed / max(self.verification_total, 1)
        }


@dataclass
class ThoughtTemplate:
    """Enhanced template with quality tracking"""
    id: str
    name: str
    category: TemplateCategory
    description: str
    content: str
    version: str
    provenance: TemplateProvenance
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'content': self.content,
            'version': self.version,
            'provenance': self.provenance.to_dict(),
            'quality_metrics': self.quality_metrics.to_dict()
        }


# ============================================================================
# PART 2: VERIFICATION TEST SUITE (Quality Assurance)
# ============================================================================

class VerificationTestSuite:
    """
    Generates and runs verification tests for templates
    This ensures quality before templates enter the meta-buffer
    """
    
    def __init__(self):
        self.test_database = self._initialize_test_database()
    
    def _initialize_test_database(self) -> Dict:
        """
        Pre-defined test cases for each category
        In production, these would be generated by LLM
        """
        return {
            TemplateCategory.MATHEMATICAL: [
                {
                    'id': 'math_001',
                    'problem': 'Solve quadratic: x² - 5x + 6 = 0',
                    'expected_steps': ['identify_coefficients', 'calculate_discriminant', 
                                      'apply_quadratic_formula', 'simplify'],
                    'expected_answer': 'x = 2 or x = 3',
                    'difficulty': 'easy'
                },
                {
                    'id': 'math_002',
                    'problem': 'Find maximum of f(x) = -2x² + 8x + 5',
                    'expected_steps': ['take_derivative', 'set_to_zero', 
                                      'solve_for_x', 'verify_maximum'],
                    'expected_answer': 'x = 2, max value = 13',
                    'difficulty': 'medium'
                },
                {
                    'id': 'math_003',
                    'problem': 'Optimize profit: P(x) = (20-x)(50+2x)',
                    'expected_steps': ['expand_function', 'take_derivative',
                                      'find_critical_points', 'check_constraints'],
                    'expected_answer': 'x = 7.5',
                    'difficulty': 'medium'
                }
            ],
            TemplateCategory.CODE_PROGRAMMING: [
                {
                    'id': 'code_001',
                    'problem': 'Sort list [3,1,4,1,5] in ascending order',
                    'expected_steps': ['identify_algorithm', 'implement_comparison',
                                      'perform_swaps', 'return_sorted'],
                    'expected_answer': '[1,1,3,4,5]',
                    'difficulty': 'easy'
                }
            ]
        }
    
    def generate_tests(self, category: TemplateCategory, count: int = 3) -> List[Dict]:
        """Get test cases for a category"""
        available = self.test_database.get(category, [])
        return available[:min(count, len(available))]
    
    def run_verification(self, template: ThoughtTemplate) -> Dict:
        """
        Run verification tests on a template
        Simulates applying template to test problems
        """
        tests = self.generate_tests(template.category)
        results = []
        
        for test in tests:
            # Simulate template application
            passed, confidence = self._apply_template_to_test(template, test)
            results.append({
                'test_id': test['id'],
                'difficulty': test['difficulty'],
                'passed': passed,
                'confidence': confidence
            })
        
        passed_count = sum(1 for r in results if r['passed'])
        total_count = len(results)
        
        # Update template metrics
        template.quality_metrics.verification_passed = passed_count
        template.quality_metrics.verification_total = total_count
        
        return {
            'template_id': template.id,
            'passed': passed_count,
            'total': total_count,
            'pass_rate': passed_count / total_count if total_count > 0 else 0,
            'details': results,
            'verdict': 'PASS' if (passed_count / total_count) >= 0.7 else 'FAIL'
        }
    
    def _apply_template_to_test(self, template: ThoughtTemplate, 
                                test: Dict) -> Tuple[bool, float]:
        """
        Simulate applying template to solve test problem
        Returns: (success, confidence)
        """
        # Simulate based on template content quality
        # In reality, this would use an LLM to apply template
        
        base_success_rate = 0.8  # Assume templates are generally good
        
        # Adjust by difficulty
        difficulty_factors = {'easy': 1.1, 'medium': 1.0, 'hard': 0.8}
        factor = difficulty_factors[test['difficulty']]
        
        # Add some randomness
        success_prob = min(base_success_rate * factor, 0.95)
        success = np.random.random() < success_prob
        
        confidence = np.random.uniform(0.7, 0.95) if success else np.random.uniform(0.3, 0.6)
        
        return success, confidence


# ============================================================================
# PART 3: MULTI-MODEL BOOTSTRAP (Cold Start Solution)
# ============================================================================

class MultiModelBootstrap:
    """
    Solves cold start by using ensemble of models
    Creates high-quality initial templates through cross-validation
    """
    
    def __init__(self, verification_suite: VerificationTestSuite):
        self.verification_suite = verification_suite
        self.model_capabilities = {
            'GPT-4': 0.95,
            'Claude-3': 0.93,
            'Llama3-70B': 0.85,
            'Llama3-8B': 0.72
        }
    
    def bootstrap_template(self, 
                          problem_examples: List[str],
                          category: TemplateCategory,
                          models: List[str] = None) -> Optional[ThoughtTemplate]:
        """
        Create template using multiple models for validation
        
        Process:
        1. Each model independently solves example problems
        2. Each model proposes a template
        3. Cross-validate templates across models
        4. Synthesize best template
        5. Run verification tests
        """
        if models is None:
            models = ['GPT-4', 'Claude-3', 'Llama3-70B']
        
        print(f"\n🔄 Bootstrapping template for {category.value}")
        print(f"   Using models: {models}")
        
        # Step 1: Collect template proposals from each model
        proposals = []
        for model in models:
            template_proposal = self._generate_template_proposal(
                model, problem_examples, category
            )
            proposals.append(template_proposal)
            print(f"   ✓ {model} generated proposal")
        
        # Step 2: Cross-validate proposals
        print(f"   Running cross-validation...")
        validation_scores = self._cross_validate_proposals(proposals, models)
        
        # Step 3: Select best proposal or synthesize
        best_template = self._synthesize_template(
            proposals, validation_scores, models, category
        )
        
        # Step 4: Run verification tests
        print(f"   Running verification tests...")
        verification_result = self.verification_suite.run_verification(best_template)
        
        print(f"   Verification: {verification_result['verdict']} "
              f"({verification_result['passed']}/{verification_result['total']})")
        
        # Only accept if passes verification
        if verification_result['verdict'] == 'PASS':
            print(f"   ✅ Template accepted with quality score: "
                  f"{best_template.quality_metrics.get_quality_score():.2f}")
            return best_template
        else:
            print(f"   ❌ Template rejected - failed verification")
            return None
    
    def _generate_template_proposal(self, model: str, 
                                    examples: List[str],
                                    category: TemplateCategory) -> Dict:
        """Simulate model generating a template proposal"""
        capability = self.model_capabilities[model]
        
        # Simulate template quality based on model capability
        quality = capability + np.random.normal(0, 0.05)
        quality = np.clip(quality, 0.5, 1.0)
        
        return {
            'model': model,
            'template_content': f"Template from {model} for {category.value}",
            'quality_estimate': quality,
            'structure_score': quality * np.random.uniform(0.9, 1.0)
        }
    
    def _cross_validate_proposals(self, proposals: List[Dict], 
                                  models: List[str]) -> Dict:
        """
        Each model validates other models' proposals
        Returns validation matrix
        """
        scores = {}
        for proposal in proposals:
            validator_scores = []
            for validator_model in models:
                if validator_model != proposal['model']:
                    # Simulate validation
                    validator_capability = self.model_capabilities[validator_model]
                    # Better validators give more accurate scores
                    score = proposal['quality_estimate'] * validator_capability
                    score += np.random.normal(0, 0.05)
                    validator_scores.append(np.clip(score, 0, 1))
            
            scores[proposal['model']] = {
                'avg_validation_score': np.mean(validator_scores),
                'std_validation_score': np.std(validator_scores)
            }
        
        return scores
    
    def _synthesize_template(self, proposals: List[Dict], 
                           validation_scores: Dict,
                           models: List[str],
                           category: TemplateCategory) -> ThoughtTemplate:
        """Synthesize best template from proposals"""
        
        # Find proposal with highest validation score
        best_model = max(validation_scores.keys(), 
                        key=lambda m: validation_scores[m]['avg_validation_score'])
        
        best_proposal = next(p for p in proposals if p['model'] == best_model)
        
        # Create template with multi-model provenance
        template_id = f"T_{category.value[:4]}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]}"
        
        template = ThoughtTemplate(
            id=template_id,
            name=f"{category.value.title()} Template (Bootstrap)",
            category=category,
            description=f"Bootstrapped template for {category.value}",
            content=best_proposal['template_content'],
            version="v1.0",
            provenance=TemplateProvenance(
                source_models=models,
                creation_date=datetime.now().isoformat(),
                validation_models=models
            )
        )
        
        return template


# ============================================================================
# PART 4: VALIDATION LAYER (Runtime Quality Assurance)
# ============================================================================

class ValidationLayer:
    """
    Validates templates at retrieval time before using them
    Prevents propagation of low-quality templates
    """
    
    def __init__(self):
        self.validation_history = []
    
    def validate_candidates(self, 
                          candidates: List[Tuple[ThoughtTemplate, float]],
                          problem: str) -> Optional[ThoughtTemplate]:
        """
        Validate each candidate template before using
        Returns first template that passes validation
        """
        print(f"\n🔍 Validating {len(candidates)} candidate templates...")
        
        for i, (template, retrieval_score) in enumerate(candidates):
            print(f"\n   Candidate {i+1}: {template.id} (score: {retrieval_score:.2f})")
            
            validation_result = self._validate_single_template(
                template, problem, retrieval_score
            )
            
            if validation_result['decision'] == 'ACCEPT':
                print(f"   ✅ ACCEPTED - Confidence: {validation_result['confidence']:.2f}")
                self.validation_history.append(validation_result)
                return template
            else:
                print(f"   ❌ REJECTED - {validation_result['reason']}")
                self.validation_history.append(validation_result)
        
        print(f"\n   ⚠️  All candidates failed validation")
        return None
    
    def _validate_single_template(self, template: ThoughtTemplate,
                                  problem: str, 
                                  retrieval_score: float) -> Dict:
        """
        Validate single template by:
        1. Generating partial solution
        2. Checking solution quality
        3. Verifying consistency with template
        """
        
        # Simulate partial solution generation
        partial_solution = self._generate_partial_solution(template, problem)
        
        # Evaluate partial solution
        solution_quality = self._evaluate_partial_solution(
            partial_solution, template, problem
        )
        
        # Consistency check
        consistency_pass = self._check_consistency(partial_solution, template)
        
        # Quality threshold
        quality_threshold = 0.7
        
        # Decision logic
        accept = (solution_quality > quality_threshold and 
                 consistency_pass and
                 template.quality_metrics.get_quality_score() > 0.6)
        
        result = {
            'template_id': template.id,
            'problem': problem[:50] + '...',
            'retrieval_score': retrieval_score,
            'partial_solution_quality': solution_quality,
            'consistency_passed': consistency_pass,
            'template_quality_score': template.quality_metrics.get_quality_score(),
            'decision': 'ACCEPT' if accept else 'REJECT',
            'confidence': solution_quality if accept else 0.5,
            'reason': 'All checks passed' if accept else self._get_failure_reason(
                solution_quality, consistency_pass, template
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _generate_partial_solution(self, template: ThoughtTemplate, 
                                   problem: str) -> str:
        """Simulate generating first few steps of solution"""
        return f"Partial solution using {template.id} for problem"
    
    def _evaluate_partial_solution(self, solution: str, 
                                   template: ThoughtTemplate,
                                   problem: str) -> float:
        """Evaluate quality of partial solution"""
        # Base quality from template
        base = template.quality_metrics.get_quality_score()
        
        # Add variance
        noise = np.random.normal(0, 0.1)
        score = np.clip(base + noise, 0, 1)
        
        return score
    
    def _check_consistency(self, solution: str, 
                          template: ThoughtTemplate) -> bool:
        """Check if solution follows template structure"""
        # Simulate consistency check
        return np.random.random() > 0.1  # 90% pass rate
    
    def _get_failure_reason(self, solution_quality: float, 
                           consistency: bool,
                           template: ThoughtTemplate) -> str:
        """Generate human-readable failure reason"""
        reasons = []
        if solution_quality <= 0.7:
            reasons.append(f"Low solution quality ({solution_quality:.2f})")
        if not consistency:
            reasons.append("Inconsistent with template structure")
        if template.quality_metrics.get_quality_score() <= 0.6:
            reasons.append(f"Low template quality score")
        return "; ".join(reasons)


# ============================================================================
# PART 5: QUALITY-AWARE META BUFFER
# ============================================================================

class QualityAwareMetaBuffer:
    """Enhanced meta-buffer with quality tracking and management"""
    
    def __init__(self, quality_threshold: float = 0.65):
        self.templates: Dict[str, ThoughtTemplate] = {}
        self.category_index: Dict[TemplateCategory, List[str]] = defaultdict(list)
        self.quality_threshold = quality_threshold
        self.verification_suite = VerificationTestSuite()
    
    def add_template(self, template: ThoughtTemplate, 
                    require_verification: bool = True) -> bool:
        """
        Add template with quality gate
        Returns: True if accepted, False if rejected
        """
        if require_verification:
            result = self.verification_suite.run_verification(template)
            if result['verdict'] != 'PASS':
                return False
        
        self.templates[template.id] = template
        self.category_index[template.category].append(template.id)
        return True
    
    def retrieve_candidates(self, 
                          problem_embedding: np.ndarray,
                          category: TemplateCategory,
                          top_k: int = 3) -> List[Tuple[ThoughtTemplate, float]]:
        """
        Retrieve top-k templates ranked by quality + similarity
        """
        candidates = []
        template_ids = self.category_index.get(category, [])
        
        for tid in template_ids:
            template = self.templates[tid]
            
            # Calculate similarity (mock)
            similarity = self._calculate_similarity(problem_embedding, template)
            
            # Get quality score
            quality = template.quality_metrics.get_quality_score()
            
            # Combined score: 60% similarity, 40% quality
            combined_score = 0.6 * similarity + 0.4 * quality
            
            # Filter by minimum quality
            if quality >= self.quality_threshold:
                candidates.append((template, combined_score))
        
        # Sort by combined score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _calculate_similarity(self, problem_emb: np.ndarray, 
                             template: ThoughtTemplate) -> float:
        """Mock similarity calculation"""
        # Simulate embedding similarity
        return np.random.uniform(0.6, 0.95)
    
    def update_template_usage(self, template_id: str, 
                            success: bool, 
                            confidence: float = 1.0):
        """Update template quality after usage"""
        if template_id in self.templates:
            self.templates[template_id].quality_metrics.add_result(success, confidence)
    
    def get_health_report(self) -> Dict:
        """Generate system health report"""
        templates_by_quality = {
            'high': [],
            'medium': [],
            'low': [],
            'declining': []
        }
        
        for template in self.templates.values():
            quality = template.quality_metrics.get_quality_score()
            trend = template.quality_metrics.get_trend()
            
            if trend == 'declining':
                templates_by_quality['declining'].append(template.id)
            elif quality >= 0.85:
                templates_by_quality['high'].append(template.id)
            elif quality >= 0.7:
                templates_by_quality['medium'].append(template.id)
            else:
                templates_by_quality['low'].append(template.id)
        
        return {
            'total_templates': len(self.templates),
            'by_category': {cat.value: len(tids) 
                          for cat, tids in self.category_index.items()},
            'by_quality': {k: len(v) for k, v in templates_by_quality.items()},
            'avg_quality': np.mean([t.quality_metrics.get_quality_score() 
                                   for t in self.templates.values()]) if self.templates else 0,
            'templates_by_quality': templates_by_quality
        }
    
    def export_templates(self, filepath: str):
        """Export templates to JSON"""
        data = {
            'templates': [t.to_dict() for t in self.templates.values()],
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_templates': len(self.templates)
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# PART 6: COMPLETE QUALITY-AWARE BOT SYSTEM
# ============================================================================

class QualityAwareBoT:
    """
    Complete BoT system with:
    1. Multi-model bootstrap (cold start solution)
    2. Quality-aware meta-buffer (quality assurance)
    3. Validation layer (runtime quality checks)
    """
    
    def __init__(self):
        self.verification_suite = VerificationTestSuite()
        self.bootstrap = MultiModelBootstrap(self.verification_suite)
        self.meta_buffer = QualityAwareMetaBuffer()
        self.validation_layer = ValidationLayer()
        self.problem_history = []
    
    def initialize_with_bootstrap(self, 
                                 categories: List[TemplateCategory] = None):
        """
        Initialize meta-buffer using multi-model bootstrap
        Solves cold start problem
        """
        if categories is None:
            categories = [TemplateCategory.MATHEMATICAL, 
                         TemplateCategory.CODE_PROGRAMMING]
        
        print("\n" + "="*70)
        print("INITIALIZING QUALITY-AWARE BOT (COLD START SOLUTION)")
        print("="*70)
        
        for category in categories:
            # Bootstrap template with multiple models
            template = self.bootstrap.bootstrap_template(
                problem_examples=["example1", "example2"],
                category=category,
                models=['GPT-4', 'Claude-3', 'Llama3-70B']
            )
            
            if template:
                self.meta_buffer.add_template(template, require_verification=False)
                print(f"\n✅ Initialized {category.value} template: {template.id}")
    
    def solve_problem(self, problem: str, 
                     category: TemplateCategory) -> Dict:
        """
        Solve problem with quality assurance at every step
        """
        print(f"\n" + "="*70)
        print(f"SOLVING PROBLEM (QUALITY-AWARE)")
        print("="*70)
        print(f"Problem: {problem}")
        print(f"Category: {category.value}")
        
        # Step 1: Distill problem (simplified)
        problem_embedding = self._distill_problem(problem)
        print(f"\n✓ Problem distilled")
        
        # Step 2: Retrieve top-k candidates with quality ranking
        candidates = self.meta_buffer.retrieve_candidates(
            problem_embedding,
            category=category,
            top_k=3
        )
        
        if not candidates:
            print(f"\n❌ No suitable templates found")
            return {'success': False, 'reason': 'No templates available'}
        
        print(f"\n✓ Retrieved {len(candidates)} candidate templates")
        for i, (t, score) in enumerate(candidates):
            print(f"   {i+1}. {t.id} - Quality: {t.quality_metrics.get_quality_score():.2f}, "
                  f"Score: {score:.2f}")
        
        # Step 3: Validate candidates (quality gate)
        selected_template = self.validation_layer.validate_candidates(
            candidates, problem
        )
        
        if not selected_template:
            print(f"\n❌ All templates failed validation")
            return {
                'success': False,
                'reason': 'All templates failed validation',
                'tried_templates': [t.id for t, _ in candidates]
            }
        
        print(f"\n✓ Selected template: {selected_template.id}")
        
        # Step 4: Generate solution (mock)
        solution = self._generate_solution(selected_template, problem)
        
        # Step 5: Verify solution and update quality
        success = self._verify_solution(solution)
        confidence = 0.9 if success else 0.4
        
        self.meta_buffer.update_template_usage(
            selected_template.id, success, confidence
        )
        
        # Record history
        self.problem_history.append({
            'problem': problem,
            'template_id': selected_template.id,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"\n{'✅ Solution correct' if success else '❌ Solution incorrect'}")
        print(f"Template quality updated: "
              f"{selected_template.quality_metrics.get_quality_score():.2f}")
        
        return {
            'success': success,
            'template_used': selected_template.id,
            'solution': solution,
            'confidence': confidence,
            'template_quality': selected_template.quality_metrics.get_quality_score()
        }
    
    def _distill_problem(self, problem: str) -> np.ndarray:
        """Mock problem distillation"""
        return np.random.randn(768)
    
    def _generate_solution(self, template: ThoughtTemplate, 
                          problem: str) -> str:
        """Mock solution generation"""
        return f"Solution using {template.id}: [steps...]"
    
    def _verify_solution(self, solution: str) -> bool:
        """Mock solution verification"""
        return np.random.random() > 0.15  # 85% success rate
    
    def get_system_status(self) -> Dict:
        """Get complete system status"""
        health = self.meta_buffer.get_health_report()
        
        recent_problems = self.problem_history[-20:] if self.problem_history else []
        recent_success_rate = (sum(1 for p in recent_problems if p['success']) / 
                             len(recent_problems) if recent_problems else 0)
        
        return {
            'meta_buffer_health': health,
            'total_problems_solved': len(self.problem_history),
            'recent_success_rate': recent_success_rate,
            'validation_history_size': len(self.validation_layer.validation_history)
        }


# ============================================================================
# PART 7: DEMONSTRATION
# ============================================================================

def run_demonstration():
    """
    Complete demonstration of quality-aware BoT addressing:
    1. Cold Start Problem (multi-model bootstrap)
    2. Quality Assurance (verification + validation)
    """
    print("\n" + "="*70)
    print("QUALITY-AWARE BUFFER OF THOUGHTS - COMPLETE DEMONSTRATION")
    print("="*70)
    
    # Initialize system
    bot = QualityAwareBoT()
    
    # PHASE 1: Bootstrap (Cold Start Solution)
    print("\n\nPHASE 1: BOOTSTRAP INITIALIZATION (Cold Start Solution)")
    print("-" * 70)
    bot.initialize_with_bootstrap([
        TemplateCategory.MATHEMATICAL,
        TemplateCategory.CODE_PROGRAMMING
    ])
    
    # PHASE 2: Solve problems with quality assurance
    print("\n\nPHASE 2: PROBLEM SOLVING WITH QUALITY ASSURANCE")
    print("-" * 70)
    
    problems = [
        ("Solve the quadratic equation: 2x² - 7x + 3 = 0", 
         TemplateCategory.MATHEMATICAL),
        ("Find maximum of profit function: P(x) = -3x² + 12x + 5",
         TemplateCategory.MATHEMATICAL),
        ("Optimize: revenue = (100-2x)(50+x) for x",
         TemplateCategory.MATHEMATICAL)
    ]
    
    results = []
    for problem, category in problems:
        result = bot.solve_problem(problem, category)
        results.append(result)
        print("\n" + "-"*70)
    
    # PHASE 3: System Health Report
    print("\n\nPHASE 3: SYSTEM HEALTH REPORT")
    print("-" * 70)
    
    status = bot.get_system_status()
    
    print(f"\n📊 Meta-Buffer Health:")
    print(f"   Total Templates: {status['meta_buffer_health']['total_templates']}")
    print(f"   Average Quality: {status['meta_buffer_health']['avg_quality']:.3f}")
    print(f"\n   Templates by Quality:")
    for quality_level, count in status['meta_buffer_health']['by_quality'].items():
        print(f"      {quality_level}: {count}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Problems Solved: {status['total_problems_solved']}")
    print(f"   Recent Success Rate: {status['recent_success_rate']:.1%}")
    print(f"   Validation Checks: {status['validation_history_size']}")
    
    # Show specific templates
    print(f"\n🎯 Template Details:")
    for template in bot.meta_buffer.templates.values():
        metrics = template.quality_metrics
        print(f"\n   {template.id}:")
        print(f"      Name: {template.name}")
        print(f"      Quality Score: {metrics.get_quality_score():.3f}")
        print(f"      Success Rate: {metrics.get_success_rate():.3f}")
        print(f"      Trend: {metrics.get_trend()}")
        print(f"      Uses: {metrics.success_count + metrics.failure_count}")
        print(f"      Verification: {metrics.verification_passed}/{metrics.verification_total}")
    
    # Export results
    print(f"\n💾 Exporting templates...")
    bot.meta_buffer.export_templates('bot_templates_export.json')
    print(f"   ✓ Exported to bot_templates_export.json")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    return bot, results, status


if __name__ == "__main__":
    # Run the complete demonstration
    bot_system, problem_results, system_status = run_demonstration()
    
    print("\n\n" + "="*70)
    print("KEY INNOVATIONS DEMONSTRATED:")
    print("="*70)
    print("""
1. ✅ COLD START SOLUTION (Multi-Model Bootstrap):
   - Templates initialized using ensemble of models (GPT-4, Claude-3, Llama3-70B)
   - Cross-validation ensures quality from the start
   - No dependency on single weak model
   
2. ✅ QUALITY ASSURANCE (Verification + Validation):
   - Verification tests before adding templates to buffer
   - Runtime validation before using templates
   - Continuous quality tracking and trend detection
   
3. ✅ SELF-CORRECTION:
   - Templates ranked by quality + similarity
   - Automatic fallback if primary template fails
   - Declining templates automatically flagged
   
4. ✅ TRANSPARENCY:
   - Full provenance tracking (which models contributed)
   - Quality metrics visible at all times
   - Validation history recorded
""")