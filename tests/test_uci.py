from flora.uci import UCIEngine


def test_uci_handshake(capsys):
    engine = UCIEngine()
    assert engine.handle("uci")
    output = capsys.readouterr().out
    assert "id name Flora" in output
    assert "uciok" in output


def test_position_startpos_with_moves():
    engine = UCIEngine()
    engine.handle("position startpos moves e2e4 e7e5")
    assert engine.board.peek().uci() == "e7e5"


def test_position_fen():
    engine = UCIEngine()
    engine.handle("position fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    assert engine.board.is_checkmate()


def test_go_depth_emits_uci_bestmove(capsys):
    engine = UCIEngine()
    engine.handle("position startpos")
    engine.handle("go depth 1")
    worker = engine.worker
    assert worker is not None
    worker.join(timeout=10)
    output = capsys.readouterr().out
    assert "info depth 1" in output
    assert "bestmove " in output
