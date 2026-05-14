#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
V6 Motion Replay — replays a recorded .txt motion file over Modbus.

Input format (same as v6_rule_based_retargeting.py / _realtime.py):
    [v0, v1, ..., v19][v0, v1, ..., v19]...
Each frame is 20 integer encoder counts already sign-flipped on thumb_00
and pinky_40 at save time, so they go straight to the motor registers.

Modbus protocol, parser, and threaded playback come from realtime_utils.
This file only owns the tkinter file-picker GUI and CLI plumbing.

Usage:
  python v6_replay_motion.py                                # GUI w/ picker
  python v6_replay_motion.py --file motion.txt              # GUI pre-loaded
  python v6_replay_motion.py --file motion.txt --no-gui --hz 50
  python v6_replay_motion.py --file motion.txt --no-gui --loop
"""

import sys
import argparse
import time
from pathlib import Path

try:
    from . import realtime_utils as rtu
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    import realtime_utils as rtu


# ----------------------------------------------------------------------
# Tkinter GUI
# ----------------------------------------------------------------------
class ReplayGUI:
    _FONT_HEADER = ('TkDefaultFont', 13, 'bold')
    _FONT_LABEL  = ('TkDefaultFont', 11)
    _FONT_BUTTON = ('TkDefaultFont', 11, 'bold')

    def __init__(self, replayer: rtu.MotionReplayer,
                 initial_file: str = None, default_hz: float = 50.0):
        import tkinter as tk
        from tkinter import filedialog
        self.tk = tk
        self.filedialog = filedialog
        self.replayer = replayer
        self.frames = []
        self.file_path = None

        self.root = tk.Tk()
        self.root.title('V6 Motion Replay')

        # File picker row
        file_row = tk.Frame(self.root)
        file_row.pack(padx=10, pady=(10, 4), fill='x')
        tk.Button(file_row, text='Select file...', font=self._FONT_BUTTON,
                  width=14, command=self._on_select_file).pack(side='left', padx=4)
        self._file_label_var = tk.StringVar(value='(no file loaded)')
        tk.Label(file_row, textvariable=self._file_label_var,
                 font=self._FONT_LABEL, anchor='w', width=50).pack(
                 side='left', padx=4, fill='x', expand=True)

        # Options row
        opt_row = tk.Frame(self.root)
        opt_row.pack(padx=10, pady=4)
        tk.Label(opt_row, text='Send Hz:', font=self._FONT_LABEL).pack(side='left')
        self._hz_var = tk.IntVar(value=int(default_hz))
        tk.Spinbox(opt_row, from_=1, to=120, increment=1,
                   textvariable=self._hz_var, width=5,
                   font=self._FONT_LABEL).pack(side='left', padx=(2, 12))
        self._loop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_row, text='Loop', variable=self._loop_var,
                       font=self._FONT_LABEL).pack(side='left')

        # Play/Pause/Stop row
        ctrl_row = tk.Frame(self.root)
        ctrl_row.pack(padx=10, pady=6)
        self._play_button = tk.Button(
            ctrl_row, text='▶ Play', font=self._FONT_BUTTON,
            width=12, fg='#080', command=self._on_play_toggle,
        )
        self._play_button.pack(side='left', padx=4)
        tk.Button(ctrl_row, text='■ Stop', font=self._FONT_BUTTON,
                  width=12, fg='#a00',
                  command=self._on_stop).pack(side='left', padx=4)

        self._progress_var = tk.StringVar(value='(no file loaded)')
        tk.Label(self.root, textvariable=self._progress_var,
                 font=self._FONT_LABEL, fg='#555').pack(padx=10, pady=(4, 10))

        if not replayer.connected:
            self._flash_warn('Modbus not connected — Play disabled until you '
                             'restart with a working port.')

        if initial_file:
            self._load(initial_file)

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._tick()

    # ----- file ops --------------------------------------------------
    def _on_select_file(self):
        path = self.filedialog.askopenfilename(
            title='Select recorded motion file',
            initialdir=str(Path(__file__).resolve().parent / 'recordings'),
            filetypes=[('Motion txt', '*.txt'), ('All files', '*')],
        )
        if path:
            self._load(path)

    def _load(self, path):
        try:
            frames = rtu.parse_motion_file(path)
        except Exception as e:
            self._file_label_var.set(f'Load failed: {e}')
            self.frames = []
            return
        if not frames:
            self._file_label_var.set(f'{Path(path).name}: no valid frames found')
            self.frames = []
            return
        self.frames = frames
        self.file_path = path
        self._file_label_var.set(f'{Path(path).name}   ({len(frames)} frames)')

    # ----- playback control -----------------------------------------
    def _on_play_toggle(self):
        rp = self.replayer
        if rp.playing:
            if rp.is_paused():
                rp.resume()
                self._play_button.config(text='⏸ Pause', fg='#a60')
            else:
                rp.pause()
                self._play_button.config(text='▶ Resume', fg='#080')
        else:
            if not self.frames:
                self._flash_warn('No file loaded.')
                return
            if not rp.connected:
                self._flash_warn('Modbus not connected.')
                return
            try:
                hz = float(self._hz_var.get())
            except (ValueError, self.tk.TclError):
                hz = 50.0
            rp.play(self.frames, target_hz=hz, loop=self._loop_var.get(),
                    on_done=self._on_playback_done)
            self._play_button.config(text='⏸ Pause', fg='#a60')

    def _on_stop(self):
        self.replayer.stop()
        self._play_button.config(text='▶ Play', fg='#080')

    def _on_playback_done(self):
        try:
            self.root.after(0, lambda:
                self._play_button.config(text='▶ Play', fg='#080'))
        except Exception:
            pass

    def _on_close(self):
        try:
            if self.replayer.playing:
                self.replayer.stop()
        finally:
            self.root.destroy()

    # ----- status / utility -----------------------------------------
    def _flash_warn(self, msg, ms=4000):
        self._progress_var.set(msg)

    def _tick(self):
        rp = self.replayer
        if rp.playing:
            state = 'PAUSED' if rp.is_paused() else 'PLAYING'
            err = f'  ({rp.send_errors} err)' if rp.send_errors > 0 else ''
            self._progress_var.set(
                f'{state}  {rp.current_idx} / {rp.total_frames}{err}'
            )
        else:
            if self.frames:
                err = (f'  (last run: {rp.send_errors} err)'
                       if rp.send_errors > 0 else '')
                self._progress_var.set(
                    f'Ready — {len(self.frames)} frames{err}'
                )
            else:
                self._progress_var.set('(no file loaded)')
        try:
            self.root.after(100, self._tick)
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Replay a recorded V6 motion .txt over Modbus.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v6_replay_motion.py
  python v6_replay_motion.py --file recordings/motion_2026-05-14_10-30-00.txt
  python v6_replay_motion.py --file motion.txt --no-gui --hz 50
  python v6_replay_motion.py --file motion.txt --no-gui --loop --hz 50
        """
    )
    parser.add_argument('--file', type=str, default=None,
                        help='Motion txt file to load.')
    parser.add_argument('--no-gui', action='store_false', dest='gui',
                        help='Headless mode — requires --file.')
    parser.add_argument('--hz', type=float, default=50.0,
                        help='Playback rate (default 50 Hz).')
    parser.add_argument('--loop', action='store_true',
                        help='Loop forever until Ctrl+C / Stop.')

    parser.add_argument('--modbus-method', choices=['rtu', 'tcp'], default='rtu')
    parser.add_argument('--modbus-port', type=str, default='/dev/ttyUSB0')
    parser.add_argument('--modbus-baud', type=int, default=2_000_000)
    parser.add_argument('--modbus-host', type=str, default='localhost')
    parser.add_argument('--modbus-tcp-port', type=int, default=502)
    parser.add_argument('--modbus-slave-id', type=int, default=1)

    args = parser.parse_args()

    if not rtu.PYMODBUS_AVAILABLE:
        print('pymodbus not installed — pip install pymodbus pyserial')
        sys.exit(1)

    replayer = rtu.MotionReplayer(
        method=args.modbus_method,
        port=args.modbus_port,
        baudrate=args.modbus_baud,
        host=args.modbus_host,
        tcp_port=args.modbus_tcp_port,
        slave_id=args.modbus_slave_id,
    )

    connected = replayer.connect()
    if not connected:
        print('[main] Modbus connect failed.')
        if not args.gui:
            sys.exit(1)

    try:
        if args.gui:
            try:
                gui = ReplayGUI(replayer,
                                initial_file=args.file,
                                default_hz=args.hz)
            except ImportError as e:
                print(f'[main] tkinter unavailable: {e}\n'
                      f'       use --no-gui --file <path> for headless mode.')
                sys.exit(1)
            gui.run()
        else:
            if not args.file:
                print('--no-gui requires --file <path>')
                sys.exit(1)
            frames = rtu.parse_motion_file(args.file)
            if not frames:
                print(f'No valid frames in {args.file}')
                sys.exit(1)
            print(f'[main] loaded {len(frames)} frames from {args.file}')
            replayer.play(frames, target_hz=args.hz, loop=args.loop)
            try:
                while replayer.playing:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                replayer.stop()
                print('\n[main] interrupted.')
    finally:
        if replayer.playing:
            replayer.stop()
            time.sleep(0.2)
        replayer.disconnect()


if __name__ == '__main__':
    main()
