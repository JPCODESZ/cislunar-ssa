# Cislunar SSA

Open-source cislunar orbit propagator for space situational awareness.

## What This Is

A Python toolkit for modeling spacecraft and debris trajectories in the Earth-Moon system using the Circular Restricted Three-Body Problem (CR3BP).

## Physics

- CR3BP equations of motion
- Lagrange point computation
- Jacobi integral and zero-velocity surfaces

## Install

```bash
pip install -r requirements.txt

**Create requirements.txt:**

```bash
cat > requirements.txt << 'EOF'
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
plotly>=5.14.0
jupyter>=1.0.0
pytest>=7.3.0
