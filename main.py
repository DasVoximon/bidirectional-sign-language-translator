#!/usr/bin/env python3
"""Launcher for the RhythmusByte sign-language application.

Usage examples:
  python main.py                 # starts Application.py by default
  python main.py --mode application
  python main.py --mode speech
  python main.py --mode speech_to_sign
"""

import argparse
import sys


def run_application():
    from Application import Application

    app = Application()
    app.root.mainloop()


def run_speech_to_sign():
    from speech_to_sign import SpeechToSignApp

    SpeechToSignApp()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch the RhythmusByte sign language app or the speech-to-sign app."
    )
    parser.add_argument(
        "--mode",
        choices=["application", "app", "speech", "speech_to_sign", "direction2"],
        default="application",
        help="Select which application to run. Default is 'application'.",
    )
    args = parser.parse_args(argv)

    if args.mode in ("application", "app"):
        run_application()
    else:
        run_speech_to_sign()


if __name__ == "__main__":
    main()
