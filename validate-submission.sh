#!/bin/bash
echo "=== ExamForge Submission Validator ==="
echo "Checking Python version..."
python3 --version

echo ""
echo "Running environment smoke test..."
python3 -c "
import sys
sys.path.insert(0, '.')
from models import ExamForgeAction, ExamForgeObservation
from server.environment import ExamForgeEnvironment
env = ExamForgeEnvironment()
obs = env.reset()
print('reset() OK — Topics:', obs.available_topics[:3])
action = ExamForgeAction(action_type='assemble_paper')
result = env.step(action)
print('step() OK — reward:', result.reward)
state = env.state
print('state OK —', state.model_dump())
print()
print('ALL CHECKS PASSED')
"
echo "=== Validation Complete ==="
