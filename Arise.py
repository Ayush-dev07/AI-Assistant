import os
import sys
os.environ["PULSE_LATENCY_MSEC"] = "60"
os.environ["ALSA_LOG_LEVEL"] = "none"
sys.stderr = open(os.devnull, "w")


from Zero_two import activate

if __name__ == "__main__":
    while True:
        activate()
        