# Team Workflow and Collaboration Guide

**🤝 Working Together for IOAI Success**

This guide outlines best practices for team collaboration during the International AI Olympiad preparation and competition.

## 🎯 Team Structure and Roles

### Recommended Team Organization
- **Team Lead**: Coordinates development, reviews solutions, manages deadlines
- **Manipulation Specialist**: Focus on grasping, pick-and-place, object manipulation
- **Navigation Expert**: Path planning, SLAM, autonomous movement
- **Vision Engineer**: Object detection, pose estimation, scene understanding
- **Integration Manager**: Combines solutions, handles multi-task scenarios

## 📋 Development Workflow

### 1. Task Assignment and Planning

```python
# Create issues for each competition challenge
# Example issue structure:
"""
Title: Implement Vision-Guided Grasping for Challenge 3
Assignee: @vision-engineer, @manipulation-specialist
Priority: High
Deadline: Week 2

Description:
- Integrate YOLO object detection with grasp planning
- Handle multiple object types (cubes, cylinders, irregular shapes)
- Achieve 95% success rate in test scenarios
- Optimize for competition time limits

Acceptance Criteria:
- [ ] Object detection accuracy > 90%
- [ ] Grasp success rate > 95%
- [ ] Execution time < 30 seconds per object
- [ ] Robust to lighting variations
- [ ] Code documented and tested
"""
```

### 2. Branch Strategy

```bash
# Main branches
main                    # Production-ready competition code
develop                 # Integration branch for testing
feature/*              # Individual feature development
hotfix/*               # Critical bug fixes during competition

# Example workflow
git checkout develop
git pull origin develop
git checkout -b feature/advanced-grasp-planning

# Work on your feature...
# Test thoroughly...

git add .
git commit -m "feat: Add advanced grasp planning with collision avoidance

- Implement grasp quality metrics
- Add collision checking for grasp poses  
- Integrate with vision pipeline
- Achieve 98% success rate on test objects

Tested on: manipulation scenarios 1-5
Performance: 25s average execution time"

git push origin feature/advanced-grasp-planning
# Create pull request to develop branch
```

### 3. Code Review Process

```python
# Example pull request template
"""
## 🎯 Competition Challenge
Which IOAI challenge does this address?
- [ ] Manipulation Challenge
- [ ] Navigation Challenge  
- [ ] Perception Challenge
- [ ] Multi-Task Integration

## 🧪 Testing Results
- Test scenarios passed: 15/15
- Performance benchmarks: 
  - Execution time: 22.3s (target: <30s)
  - Success rate: 96.7% (target: >95%)
  - Memory usage: 124MB (limit: 256MB)

## 🔍 Changes Made
- Brief description of implementation
- Key algorithmic improvements
- Performance optimizations

## 🤔 Review Checklist
- [ ] Code follows team conventions
- [ ] All tests pass
- [ ] Performance meets competition requirements
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Error handling implemented
"""
```

## 🚀 Shared Development Patterns

### 1. Consistent Environment Setup

```python
# shared_config.py - Team-wide configuration
class IOAITeamConfig:
    """Shared configuration for team development"""
    
    # Competition constraints
    MAX_EXECUTION_TIME = 60.0  # seconds
    MAX_MEMORY_USAGE = 512     # MB
    TARGET_SUCCESS_RATE = 0.95 # 95%
    
    # Robot configuration
    ROBOT_CONFIG = {
        "enabled_modules": [
            "chassis", "left_arm", "right_arm", 
            "left_gripper", "right_gripper", "head",
            "front_head_camera", "left_wrist_camera", "right_wrist_camera"
        ],
        "control_frequency": 100,  # Hz
        "physics_timestep": 0.001  # seconds
    }
    
    # Vision settings
    VISION_CONFIG = {
        "detection_threshold": 0.8,
        "max_objects_per_frame": 10,
        "camera_resolution": (640, 480)
    }

# Usage in team code
from shared_config import IOAITeamConfig

class YourSolution:
    def __init__(self):
        self.config = IOAITeamConfig()
        self.max_time = self.config.MAX_EXECUTION_TIME
```

### 2. Standardized Callback Pattern

```python
# team_callback_base.py - Base class for all team solutions
import time
from abc import ABC, abstractmethod

class IOAITeamCallback(ABC):
    """Base class for all team competition callbacks"""
    
    def __init__(self, name, max_execution_time=60.0):
        self.name = name
        self.max_execution_time = max_execution_time
        self.start_time = None
        self.execution_stats = {
            "success_rate": 0.0,
            "avg_execution_time": 0.0,
            "total_attempts": 0,
            "successful_attempts": 0
        }
    
    def __call__(self):
        """Main callback execution with timing and error handling"""
        if self.start_time is None:
            self.start_time = time.time()
        
        try:
            # Check time constraint
            elapsed = time.time() - self.start_time
            if elapsed > self.max_execution_time:
                self.on_timeout()
                return
            
            # Execute the main algorithm
            success = self.execute()
            
            # Update statistics
            self.update_stats(success, elapsed)
            
        except Exception as e:
            print(f"Error in {self.name}: {e}")
            self.on_error(e)
    
    @abstractmethod
    def execute(self) -> bool:
        """Implement your competition algorithm here"""
        pass
    
    def on_timeout(self):
        """Handle timeout situations"""
        print(f"{self.name}: Execution timeout after {self.max_execution_time}s")
    
    def on_error(self, error):
        """Handle errors gracefully"""
        print(f"{self.name}: Error occurred - {error}")
    
    def update_stats(self, success, execution_time):
        """Update performance statistics"""
        self.execution_stats["total_attempts"] += 1
        if success:
            self.execution_stats["successful_attempts"] += 1
        
        self.execution_stats["success_rate"] = (
            self.execution_stats["successful_attempts"] / 
            self.execution_stats["total_attempts"]
        )
        
        # Update average execution time
        n = self.execution_stats["total_attempts"]
        old_avg = self.execution_stats["avg_execution_time"]
        self.execution_stats["avg_execution_time"] = (
            (old_avg * (n-1) + execution_time) / n
        )

# Example team implementation
class GraspingChallengeSolution(IOAITeamCallback):
    def __init__(self, env):
        super().__init__("GraspingChallenge", max_execution_time=45.0)
        self.env = env
    
    def execute(self) -> bool:
        """Implement grasp challenge solution"""
        # Your team's grasping algorithm
        objects = self.env.detect_objects()
        if not objects:
            return False
        
        target = self.select_best_target(objects)
        success = self.env.grasp_object(target)
        
        if success:
            self.env.place_object(target_location=[1.0, 0.0, 0.1])
        
        return success
    
    def select_best_target(self, objects):
        """Team-specific target selection strategy"""
        # Implement your object selection logic
        return objects[0]  # Placeholder
```

### 3. Shared Testing Framework

```python
# team_testing.py - Standardized testing for all solutions
import json
import time
from typing import List, Dict, Any

class IOAITeamTester:
    """Standardized testing framework for team solutions"""
    
    def __init__(self):
        self.test_results = []
    
    def run_test_suite(self, solution_callback, test_scenarios: List[Dict]):
        """Run a complete test suite on a solution"""
        print(f"🧪 Running test suite for {solution_callback.name}")
        print(f"📋 Testing {len(test_scenarios)} scenarios")
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\n🎯 Test {i+1}/{len(test_scenarios)}: {scenario['name']}")
            
            # Setup scenario
            self.setup_scenario(scenario)
            
            # Run test
            start_time = time.time()
            success = self.run_single_test(solution_callback)
            execution_time = time.time() - start_time
            
            # Record results
            result = {
                "scenario": scenario['name'],
                "success": success,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            self.test_results.append(result)
            
            # Print immediate feedback
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {execution_time:.2f}s")
        
        self.print_summary()
        return self.test_results
    
    def setup_scenario(self, scenario: Dict):
        """Setup a specific test scenario"""
        # Implement scenario setup logic
        pass
    
    def run_single_test(self, callback) -> bool:
        """Run a single test case"""
        try:
            return callback.execute()
        except Exception as e:
            print(f"Test failed with error: {e}")
            return False
    
    def print_summary(self):
        """Print test suite summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        avg_time = sum(r['execution_time'] for r in self.test_results) / total_tests
        
        print(f"\n📊 Test Suite Summary")
        print(f"{'='*50}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Avg Execution Time: {avg_time:.2f}s")
        
        if success_rate >= 0.95:
            print("🏆 COMPETITION READY! Great job team!")
        elif success_rate >= 0.80:
            print("⚠️  Good progress, but needs improvement")
        else:
            print("🔧 Needs significant work before competition")

# Example usage for team testing
def run_team_tests():
    """Run all team solutions through standardized tests"""
    
    # Define test scenarios
    test_scenarios = [
        {
            "name": "Basic Object Grasp",
            "difficulty": "easy",
            "objects": ["cube_red", "cylinder_blue"],
            "target_location": [1.0, 0.0, 0.1]
        },
        {
            "name": "Cluttered Scene",
            "difficulty": "medium", 
            "objects": ["cube_red", "cube_blue", "cylinder_green", "sphere_yellow"],
            "obstacles": ["wall_left", "wall_right"],
            "target_location": [1.5, 0.0, 0.1]
        },
        {
            "name": "Moving Target",
            "difficulty": "hard",
            "objects": ["moving_cube"],
            "dynamics": True,
            "target_location": [2.0, 1.0, 0.1]
        }
    ]
    
    # Test each team member's solution
    tester = IOAITeamTester()
    
    # Test manipulation solution
    from your_team_solutions import GraspingSolution
    grasp_solution = GraspingSolution()
    tester.run_test_suite(grasp_solution, test_scenarios)
    
    # Save results for team review
    with open('team_test_results.json', 'w') as f:
        json.dump(tester.test_results, f, indent=2)

if __name__ == "__main__":
    run_team_tests()
```

## 🏆 Competition Day Workflow

### Pre-Competition Checklist
```bash
# 1. Update from main branch
git checkout main
git pull origin main

# 2. Run full test suite
python team_testing.py

# 3. Check performance benchmarks
python benchmark_solutions.py

# 4. Verify all dependencies
pip freeze > competition_requirements.txt

# 5. Create competition build
python build_competition_package.py
```

### During Competition
- **Assign roles**: One person drives, others support
- **Time management**: Track remaining time for each challenge
- **Quick debugging**: Use shared debugging utilities
- **Backup solutions**: Have fallback approaches ready
- **Team communication**: Clear, concise updates

### Post-Competition Review
```python
# Generate team performance report
def generate_team_report():
    """Analyze team performance across all challenges"""
    
    results = load_competition_results()
    
    report = {
        "overall_score": calculate_total_score(results),
        "challenge_breakdown": analyze_by_challenge(results),
        "time_utilization": analyze_time_usage(results),
        "success_factors": identify_strengths(results),
        "improvement_areas": identify_weaknesses(results)
    }
    
    save_report(report)
    print("📊 Team performance report generated!")
```

## 🤖 Communication Protocols

### Daily Standups
- **What did you accomplish yesterday?**
- **What will you work on today?**
- **Any blockers or help needed?**
- **Integration points with other team members?**

### Code Reviews
- **Focus on competition requirements**
- **Check performance constraints**
- **Verify error handling**
- **Ensure code readability**

### Integration Testing
- **Weekly integration sessions**
- **Test multi-component solutions**
- **Verify interfaces between modules**
- **Performance testing on full scenarios**

---

## 🎯 Success Metrics for Team

- **Code Quality**: All PRs reviewed and approved
- **Test Coverage**: >90% success rate on test scenarios  
- **Performance**: Solutions meet time/memory constraints
- **Integration**: Seamless collaboration between components
- **Documentation**: Clear, helpful documentation for teammates

**Remember**: We're stronger together! 🤝 Let's build solutions that showcase our team's collaborative excellence at IOAI 2025! 🏆
