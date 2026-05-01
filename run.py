"""
Entry point. Runs the simulation then analysis.
Usage: python run.py
"""

import simulate
import analyze

if __name__ == "__main__":
    simulate.run()
    df = analyze.load()
    analyze.print_stats(df)
    analyze.plot(df)
