'''In-memory transcript capture for Auto setup before its output path exists.

The Auto controller cannot open its final terminal log until it has parsed the
Header and resolved the run output directory. This helper preserves the
important, non-secret setup transcript that occurs in that interval, then
allows the controller to prepend it to the approved run's terminal log.
'''

import io


class _TranscriptTee:
    '''Mirrors one stream to its original destination and a text buffer.'''

    def __init__(self, stream, buffer):
        self.stream = stream
        self.buffer = buffer

    def write(self, message):
        self.stream.write(message)
        self.buffer.write(message)
        self.buffer.flush()

    def flush(self):
        self.stream.flush()
        self.buffer.flush()

    def isatty(self):
        return self.stream.isatty()


class AutoPreOutputTranscript:
    '''Captures Auto Header/setup output until a run log can be opened.

    The caller supplies the ``sys``-like module so this class can be tested
    without changing the process-wide streams. It never creates files.
    '''

    def __init__(self, sys_module):
        self._sys_module = sys_module
        self._buffer = io.StringIO()
        self._original_stdout = None
        self._original_stderr = None
        self._stdout_tee = None
        self._stderr_tee = None

    @property
    def active(self):
        '''Whether this transcript currently owns the supplied streams.'''
        return self._stdout_tee is not None

    def start(self):
        '''Begins mirroring stdout and stderr into the in-memory transcript.'''
        if self.active:
            return

        self._original_stdout = self._sys_module.stdout
        self._original_stderr = self._sys_module.stderr
        self._stdout_tee = _TranscriptTee(
            self._original_stdout,
            self._buffer
        )
        self._stderr_tee = _TranscriptTee(
            self._original_stderr,
            self._buffer
        )
        self._sys_module.stdout = self._stdout_tee
        self._sys_module.stderr = self._stderr_tee

    def consume(self):
        '''Restores streams and returns the buffered transcript exactly once.'''
        transcript = self._buffer.getvalue()
        self._restore_streams()
        self._buffer.close()
        return transcript

    def discard(self):
        '''Restores streams without returning or persisting captured output.'''
        self._restore_streams()
        self._buffer.close()

    def _restore_streams(self):
        '''Restores only streams still owned by this transcript instance.'''
        if self._stdout_tee is not None:
            if self._sys_module.stdout is self._stdout_tee:
                self._sys_module.stdout = self._original_stdout

            if self._sys_module.stderr is self._stderr_tee:
                self._sys_module.stderr = self._original_stderr

        self._stdout_tee = None
        self._stderr_tee = None
        self._original_stdout = None
        self._original_stderr = None
