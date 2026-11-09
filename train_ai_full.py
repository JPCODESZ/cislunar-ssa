#!/usr/bin/env python
"""Run the supported trainer.

There is one training path. This script calls ``train_ai.main`` so a
checkout that still invokes ``train_ai_full.py`` writes the same
checkpoint ``eval_ai.py`` loads.
"""

from train_ai import main


if __name__ == "__main__":
    main()
