import pytest
from ska_tango_testing.integration import TangoEventTracer


@pytest.fixture
def tracer():
    instance = TangoEventTracer()
    yield instance
    instance.unsubscribe_all()
