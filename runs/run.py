#!/usr/bin/env python
"""
Minimal run script
"""
import imagine as img
import sys, logging, os
sys.path.append('../')

if __name__ == '__main__':
    # Checks command line arguments
    cmd, args = sys.argv[0], sys.argv[1:]
    if len(args)==0:
        print('Usage: {} RUN_DIRECTORY'.format(cmd))
        exit()

    run_directory = args[0]

    # Sets up logging
    logging.basicConfig(
      filename=os.path.join(run_directory, 'imagine.log'),
      level=logging.INFO)

    # Load pipeline
    pipeline = img.load_pipeline(run_directory)
    # Run pipeline
    pipeline()


