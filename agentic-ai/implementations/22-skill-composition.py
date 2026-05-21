"""
Auto-generated from 22-skill-composition.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Skill Composition
# Objectives: Modular skills, orchestration patterns, composition strategies, error handling in skill chains
# ======================================================================

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Level 1: Basic Sequential Skill Composition

@dataclass
class SkillResult:
    success: bool
    data: Any
    error: str = None

class Skill:
    """Base skill class."""
    def __init__(self, name: str):
        self.name = name
    
    async def execute(self, input_data: Any) -> SkillResult:
        raise NotImplementedError

class SearchSkill(Skill):
    def __init__(self):
        super().__init__('search')
    
    async def execute(self, query: str) -> SkillResult:
        await asyncio.sleep(0.2)  # Simulate API call
        results = [
            {'title': f'Result 1 for {query}', 'snippet': 'Info...'},
            {'title': f'Result 2 for {query}', 'snippet': 'More info...'}
        ]
        return SkillResult(success=True, data=results)

class AnalyzeSkill(Skill):
    def __init__(self):
        super().__init__('analyze')
    
    async def execute(self, results: List[Dict]) -> SkillResult:
        await asyncio.sleep(0.15)
        analysis = {
            'count': len(results),
            'insight': f'Found {len(results)} key points',
            'confidence': 0.85
        }
        return SkillResult(success=True, data=analysis)

class SummarizeSkill(Skill):
    def __init__(self):
        super().__init__('summarize')
    
    async def execute(self, analysis: Dict) -> SkillResult:
        await asyncio.sleep(0.1)
        summary = f'Summary: {analysis["insight"]} (confidence: {analysis["confidence"]}%)'
        return SkillResult(success=True, data=summary)

class BasicCompositor:
    def __init__(self, skills: List[Skill]):
        self.skills = {s.name: s for s in skills}
    
    async def execute_pipeline(self, initial_input: Any, pipeline: List[str]) -> SkillResult:
        """Execute skills sequentially."""
        print(f'Pipeline: {"->".join(pipeline)}')
        current = initial_input
        
        for skill_name in pipeline:
            skill = self.skills[skill_name]
            result = await skill.execute(current)
            
            if not result.success:
                print(f'❌ {skill_name} failed: {result.error}')
                return result
            
            print(f'✓ {skill_name}: {str(result.data)[:50]}...')
            current = result.data
        
        return SkillResult(success=True, data=current)

# Test Level 1
async def test_level1():
    skills = [SearchSkill(), AnalyzeSkill(), SummarizeSkill()]
    compositor = BasicCompositor(skills)
    
    result = await compositor.execute_pipeline(
        initial_input='AI trends 2024',
        pipeline=['search', 'analyze', 'summarize']
    )
    print(f'\nFinal result: {result.data}\n')

await test_level1()


# Level 2: Advanced Composition with Conditionals, Error Handling, and Fallbacks

class ConditionalCompositor:
    def __init__(self, skills: Dict[str, Skill], fallbacks: Dict[str, List[str]] = None):
        self.skills = skills
        self.fallbacks = fallbacks or {}
    
    async def execute_with_fallback(self, skill_name: str, input_data: Any, max_retries: int = 2) -> SkillResult:
        """Execute skill with retry and fallback support."""
        print(f'  Executing {skill_name}...')
        
        # Try primary skill
        skill = self.skills.get(skill_name)
        if not skill:
            return SkillResult(success=False, error=f'Skill {skill_name} not found')
        
        for attempt in range(max_retries):
            try:
                result = await skill.execute(input_data)
                if result.success:
                    return result
                print(f'    ⚠️  Attempt {attempt + 1} failed: {result.error}')
            except Exception as e:
                print(f'    ⚠️  Attempt {attempt + 1} error: {str(e)[:40]}')
        
        # Try fallback skills
        fallback_list = self.fallbacks.get(skill_name, [])
        for fallback_name in fallback_list:
            print(f'    → Trying fallback: {fallback_name}')
            fallback = self.skills.get(fallback_name)
            if fallback:
                result = await fallback.execute(input_data)
                if result.success:
                    print(f'    ✓ Fallback succeeded')
                    return result
        
        return SkillResult(success=False, error=f'All attempts failed for {skill_name}')
    
    async def conditional_pipeline(self, input_data: Any, condition: str, pipelines: Dict[str, List[str]]) -> SkillResult:
        """Execute different pipeline based on condition."""
        pipeline = pipelines.get(condition)
        if not pipeline:
            return SkillResult(success=False, error=f'Unknown condition: {condition}')
        
        print(f'Condition: {condition} → Pipeline: {"->".join(pipeline)}')
        current = input_data
        
        for skill_name in pipeline:
            result = await self.execute_with_fallback(skill_name, current)
            if not result.success:
                print(f'  ✗ Failed at {skill_name}')
                return result
            print(f'  ✓ {skill_name} succeeded')
            current = result.data
        
        return SkillResult(success=True, data=current)

# Test Level 2
async def test_level2():
    skills = {
        'search': SearchSkill(),
        'search_cache': SearchSkill(),  # Fallback
        'analyze': AnalyzeSkill(),
        'summarize': SummarizeSkill()
    }
    fallbacks = {
        'search': ['search_cache']
    }
    
    compositor = ConditionalCompositor(skills, fallbacks)
    
    pipelines = {
        'research': ['search', 'analyze', 'summarize'],
        'quick': ['search', 'summarize'],
        'deep': ['search', 'analyze']
    }
    
    # Test different conditions
    print('Level 2 - Conditional Pipelines:\n')
    result = await compositor.conditional_pipeline(
        input_data='machine learning',
        condition='research',
        pipelines=pipelines
    )
    print(f'Result: {result.data}\n')

await test_level2()


# Example 1: Parallel Skills with Synchronization

class ParallelCompositor:
    def __init__(self, skills: Dict[str, Skill]):
        self.skills = skills
    
    async def parallel_execution(self, input_data: Any, skill_names: List[str]) -> Dict[str, SkillResult]:
        """Execute multiple skills in parallel (no dependencies)."""
        print(f'Executing in parallel: {skill_names}')
        
        tasks = [
            self.skills[name].execute(input_data)
            for name in skill_names
        ]
        results = await asyncio.gather(*tasks)
        
        return {name: result for name, result in zip(skill_names, results)}
    
    async def sequential_with_parallel(self, input_data: Any, pipeline: List[Any]) -> SkillResult:
        """Execute pipeline with some sequential, some parallel steps.
        
        Example pipeline:
        ['search']  # Sequential
        [['analyze', 'extract']]  # Parallel (both use search result)
        ['summarize']  # Sequential (uses both analyze and extract)
        """
        current = input_data
        
        for step in pipeline:
            if isinstance(step, str):
                # Sequential skill
                result = await self.skills[step].execute(current)
                if not result.success:
                    return result
                current = result.data
                print(f'✓ {step}')
            elif isinstance(step, list):
                # Parallel skills
                results = await self.parallel_execution(current, step)
                print(f'✓ Parallel: {step}')
                # Combine results from parallel skills
                current = {name: r.data for name, r in results.items()}
        
        return SkillResult(success=True, data=current)

# Test Example 1
async def test_example1():
    skills = {
        'search': SearchSkill(),
        'analyze': AnalyzeSkill(),
        'summarize': SummarizeSkill()
    }
    
    compositor = ParallelCompositor(skills)
    
    print('Example 1 - Parallel Skills:\n')
    
    # Simple parallel
    results = await compositor.parallel_execution('AI', ['search', 'analyze'])
    print(f'Results: {list(results.keys())}\n')

await test_example1()


# Example 2: Skills with State and Context Passing

class ContextualSkill(Skill):
    """Skill that reads and writes to shared context."""
    
    async def execute_with_context(self, input_data: Any, context: Dict[str, Any]) -> SkillResult:
        """Skills can read/write context for state passing."""
        raise NotImplementedError

class EnhancedSearchSkill(ContextualSkill):
    def __init__(self):
        super().__init__('search')
    
    async def execute_with_context(self, query: str, context: Dict) -> SkillResult:
        await asyncio.sleep(0.1)
        results = [{'title': f'Result for {query}', 'snippet': 'Info'}]
        
        # Write to context
        context['last_search_query'] = query
        context['search_result_count'] = len(results)
        
        return SkillResult(success=True, data=results)

class EnhancedAnalyzeSkill(ContextualSkill):
    def __init__(self):
        super().__init__('analyze')
    
    async def execute_with_context(self, results: List, context: Dict) -> SkillResult:
        await asyncio.sleep(0.1)
        
        # Read from context (can use previous results)
        query = context.get('last_search_query', 'unknown')
        
        analysis = {
            'items': len(results),
            'query': query,
            'relevance': 0.9
        }
        
        # Write to context
        context['analysis_done'] = True
        context['relevance_score'] = analysis['relevance']
        
        return SkillResult(success=True, data=analysis)

class ContextualCompositor:
    def __init__(self, skills: Dict[str, ContextualSkill]):
        self.skills = skills
        self.context = {}
    
    async def execute_with_context(self, input_data: Any, pipeline: List[str]) -> SkillResult:
        """Execute pipeline with shared context."""
        print(f'Executing pipeline with context: {"->" .join(pipeline)}')
        current = input_data
        
        for skill_name in pipeline:
            skill = self.skills[skill_name]
            result = await skill.execute_with_context(current, self.context)
            
            if not result.success:
                return result
            
            print(f'  ✓ {skill_name}, context: {self.context}')
            current = result.data
        
        return SkillResult(success=True, data=current)

# Test Example 2
async def test_example2():
    skills = {
        'search': EnhancedSearchSkill(),
        'analyze': EnhancedAnalyzeSkill()
    }
    
    compositor = ContextualCompositor(skills)
    
    print('\nExample 2 - Context Passing:\n')
    result = await compositor.execute_with_context(
        input_data='climate science',
        pipeline=['search', 'analyze']
    )
    print(f'\nFinal context state: {compositor.context}')

await test_example2()


# Example 3: Loop-Based Skill Composition (Iterative Skills)

class IterativeCompositor:
    def __init__(self, skills: Dict[str, Skill]):
        self.skills = skills
    
    async def loop_until_condition(self, 
                                   initial_input: Any, 
                                   skill_name: str, 
                                   condition_fn,
                                   max_iterations: int = 5) -> SkillResult:
        """Execute same skill repeatedly until condition is met."""
        print(f'Looping skill "{skill_name}" until condition met (max {max_iterations} iterations)')
        
        current = initial_input
        iteration = 0
        
        while iteration < max_iterations:
            skill = self.skills[skill_name]
            result = await skill.execute(current)
            
            if not result.success:
                return result
            
            iteration += 1
            print(f'  Iteration {iteration}: {str(result.data)[:40]}...')
            
            # Check condition
            if condition_fn(result.data):
                print(f'  ✓ Condition met after {iteration} iterations')
                return SkillResult(success=True, data=result.data)
            
            current = result.data
        
        return SkillResult(success=False, error=f'Max iterations ({max_iterations}) reached')
    
    async def refine_pipeline(self,
                             initial_input: Any,
                             pipeline: List[str],
                             refine_skill: str,
                             quality_threshold: float = 0.9) -> SkillResult:
        """Execute pipeline, then iteratively refine using one skill."""
        print(f'Pipeline with refinement: {"->" .join(pipeline)} → refine')
        
        # Initial pipeline
        current = initial_input
        for skill_name in pipeline:
            skill = self.skills[skill_name]
            result = await skill.execute(current)
            if not result.success:
                return result
            print(f'  ✓ {skill_name}')
            current = result.data
        
        # Refinement loop
        iteration = 0
        while iteration < 3:
            quality = 0.7 + (iteration * 0.1)  # Simulated quality improvement
            if quality >= quality_threshold:
                print(f'  ✓ Quality {quality:.1%} >= threshold {quality_threshold:.1%}')
                break
            
            skill = self.skills[refine_skill]
            result = await skill.execute(current)
            if not result.success:
                break
            iteration += 1
            print(f'  Refinement {iteration}: quality {quality:.1%}')
            current = result.data
        
        return SkillResult(success=True, data=current)

# Test Example 3
async def test_example3():
    skills = {
        'search': SearchSkill(),
        'refine': AnalyzeSkill()
    }
    
    compositor = IterativeCompositor(skills)
    
    print('\nExample 3 - Iterative Skills:\n')
    
    # Refine until high quality
    result = await compositor.refine_pipeline(
        initial_input='data science',
        pipeline=['search'],
        refine_skill='refine',
        quality_threshold=0.95
    )
    print(f'Final result quality verified')

await test_example3()


# ======================================================================
# ## Key Takeaways
# **Skill Composition Patterns:**
# 1. Sequential - Skill A → B → C (most common)
# 2. Conditional - If X, run A, else run B
# ======================================================================
