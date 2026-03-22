"""
Entry point for the face tracker pipeline.
Handles startup recovery, thread orchestration, and graceful shutdown.
"""

import signal
import sys
import argparse
import logging
from modules.config import Config
from modules.logger import FaceTrackerLogger
from modules.database import Database
from capture import Pipeline

pipeline_instance = None
db_instance = None
session_id = None
logger_instance = None

def signal_handler(sig, frame):
    logging.info("Graceful shutdown triggered...")
    if pipeline_instance:
        pipeline_instance.stop()
        pipeline_instance.t2.join(timeout=10)
        pipeline_instance.t4.join(timeout=10)
    
    if db_instance and session_id:
        db_instance.end_session(session_id)
        final_count = db_instance.get_unique_visitor_count(session_id)
        logging.info(f"Session ended. Total unique visitors: {final_count}")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--source", help="Override video source")
    args = parser.parse_args()

    config = Config(args.config)
    if args.source:
        config.video.source = args.source

    global dict_snapshot
    import json
    with open(args.config, 'r') as f:
        dict_snapshot = json.load(f)

    global logger_instance
    logger_instance = FaceTrackerLogger(config.system.log_dir)
    
    global db_instance
    db_instance = Database(config)
    db_instance.run_startup_recovery()
    
    global session_id
    session_id = db_instance.register_session(config.video.source, dict_snapshot)

    signal.signal(signal.SIGINT, signal_handler)

    global pipeline_instance
    pipeline_instance = Pipeline(config, db_instance, logger_instance, session_id)
    
    try:
        pipeline_instance.start()
        # Keep main thread alive
        while not pipeline_instance.stop_event.is_set():
            import time
            time.sleep(1)
        
        logging.info("Pipeline stopped. Finalizing session...")
        # T2 already joined T3 internally. Now wait for T4 to finish all DB writes.
        pipeline_instance.t4.join(timeout=30)  # wait for T4 to drain io_queue, but don't hang forever
        db_instance.end_session(session_id)
        final_count = db_instance.get_unique_visitor_count(session_id)
        logging.info(f"Session ended cleanly. Total unique visitors: {final_count}")
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
