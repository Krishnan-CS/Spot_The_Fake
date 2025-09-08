// App.jsx
import React, { useEffect, useMemo, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  Container,
  Row,
  Col,
  Card,
  Badge,
  Button,
  Spinner,
  ProgressBar,
  ListGroup,
} from "react-bootstrap";

function Indicator({ title, status }) {
  return (
    <Badge bg={status ? "success" : "danger"} className="me-2 mb-2">
      {status ? "✔" : "✖"} {title.replace(/_/g, " ")}
    </Badge>
  );
}

export default function App() {
  const [results, setResults] = useState([]);
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    fetch("http://localhost:5000/api/results")
      .then((res) => res.json())
      .then((data) => setResults(data));
  }, []);

  const containerClass = darkMode ? "bg-dark text-light" : "bg-light text-dark";

  const USE_FLASK_IMAGES = false;
  const imageSrc = (name) =>
    USE_FLASK_IMAGES
      ? `http://localhost:5000/screenshots/${encodeURIComponent(name)}`
      : `/Screenshots/${encodeURIComponent(name)}`;

  const analytics = useMemo(() => {
    const total = results.length;
    let phish = 0;
    let legit = 0;
    let clones = 0;
    let confSum = 0;
    let confCount = 0;
    const indicatorTrueCounts = {};

    for (const r of results) {
      const isPhish = r.summary_judgment === "Phish";
      if (isPhish) phish++;
      else legit++;

      if (r.similarity_score !== undefined && r.similarity_score !== null && r.similarity_score !== "") {
        clones++;
      }

      const c = Number(r.confidence_score);
      if (!Number.isNaN(c)) {
        confSum += c;
        confCount++;
      }

      if (r.phishing_indicators) {
        for (const [key, val] of Object.entries(r.phishing_indicators)) {
          if (val && val.status) {
            indicatorTrueCounts[key] = (indicatorTrueCounts[key] || 0) + 1;
          }
        }
      }
    }

    const topIndicators = Object.entries(indicatorTrueCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6);

    const avgConfidence = confCount ? (confSum / confCount) : 0;
    const phishPct = total ? Math.round((phish / total) * 100) : 0;
    const legitPct = total ? Math.round((legit / total) * 100) : 0;
    const clonePct = total ? Math.round((clones / total) * 100) : 0;

    return {
      total,
      phish,
      legit,
      clones,
      avgConfidence,
      phishPct,
      legitPct,
      clonePct,
      topIndicators,
    };
  }, [results]);

  return (
    <div
      className={`min-vh-100 py-4 ${containerClass}`}
      style={{
        transition: "0.3s ease-in-out",
        background: darkMode
          ? "linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%)"
          : "linear-gradient(135deg, #f7faff 0%, #eef6ff 100%)",
      }}
    >
      <Container fluid>
        <div className="d-flex justify-content-between align-items-center mb-4 px-2 px-lg-3">
          <h1 className="fw-bold mb-0">🛡️ Phishing Detection Dashboard</h1>
          <Button
            variant={darkMode ? "light" : "dark"}
            onClick={() => setDarkMode(!darkMode)}
          >
            Toggle {darkMode ? "Light" : "Dark"} Mode
          </Button>
        </div>

        {results.length === 0 ? (
          <div className="text-center">
            <Spinner animation="border" variant={darkMode ? "light" : "dark"} />
            <p className="mt-3">Loading results...</p>
          </div>
        ) : (
          <Row className="gx-3 gy-4 px-2 px-lg-3">
            <Col xs={12} lg={9}>
              <Row xs={1} md={2} className="g-4">
                {results.map((item, idx) => {
                  const isPhish = item.summary_judgment === "Phish";
                  const hasSimilarity = item.similarity_score !== undefined && item.similarity_score !== null && item.similarity_score !== "";

                  const cardStyle = {
                    backgroundColor: isPhish ? "#ffe5e5" : "#e6ffed",
                    borderRadius: "12px",
                    overflow: "hidden",
                    boxShadow: isPhish ? "0 8px 16px rgba(220,53,69,.15)" : "0 8px 16px rgba(25,135,84,.15)",
                    transition: "transform .2s ease, box-shadow .2s ease",
                  };

                  const bannerStyle = {
                    background: isPhish ? "linear-gradient(90deg, #dc3545 0%, #ff6b6b 100%)" : "linear-gradient(90deg, #198754 0%, #48c78e 100%)",
                    color: "#fff",
                    fontWeight: "600",
                    textAlign: "center",
                    padding: "0.5rem",
                    fontSize: "1rem",
                  };

                  const cloneBannerStyle = {
                    background: "linear-gradient(90deg, #6f42c1 0%, #9d6bff 100%)",
                    color: "#fff",
                    fontWeight: "600",
                    textAlign: "center",
                    padding: "0.4rem",
                    fontSize: "0.95rem",
                  };

                  return (
                    <Col key={idx}>
                      <Card style={cardStyle} className="h-100">
                        <div style={bannerStyle}>
                          {isPhish ? "🚩 PHISH" : "✅ LEGITIMATE"}
                        </div>

                        {hasSimilarity && isPhish && (
                          <div style={cloneBannerStyle}>🌀 CLONE DETECTED</div>
                        )}

                        {item.screenshot && (
                          <Card.Img
                            variant="top"
                            src={imageSrc(item.screenshot)}
                            alt={item.screenshot}
                            style={{ maxHeight: "250px", objectFit: "cover" }}
                            onError={(e) => {
                              console.warn("Image failed to load:", e.currentTarget.src);
                              e.currentTarget.style.display = "none";
                            }}
                          />
                        )}

                        <Card.Body>
                          <div className="mb-2">
                            <strong>Confidence:</strong> {(Number(item.confidence_score) * 100).toFixed(0)}%
                          </div>

                          <div className="mt-2">
                            <h6 className="fw-bold">Phishing Indicators</h6>
                            <div className="d-flex flex-wrap">
                              {Object.entries(item.phishing_indicators).map(([key, val], i) => (
                                <Indicator key={i} title={key} status={val.status} />
                              ))}
                            </div>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  );
                })}
              </Row>
            </Col>

            <Col xs={12} lg={3}>
              <div className="position-sticky" style={{ top: "1rem", maxHeight: "calc(100vh - 2rem)", overflowY: "auto" }}>
                <Card className="mb-3 shadow-sm">
                  <Card.Body>
                    <Card.Title className="mb-3">Overview</Card.Title>
                    <Row className="g-2">
                      <Col xs={6}><div className="p-2 rounded bg-light text-center"><div className="fw-semibold">Sites</div><div className="fs-4">{analytics.total}</div></div></Col>
                      <Col xs={6}><div className="p-2 rounded bg-light text-center"><div className="fw-semibold">Clones</div><div className="fs-4">{analytics.clones}</div></div></Col>
                      <Col xs={6}><div className="p-2 rounded bg-danger bg-opacity-10 text-center"><div className="fw-semibold text-danger">Phish</div><div className="fs-4 text-danger">{analytics.phish}</div></div></Col>
                      <Col xs={6}><div className="p-2 rounded bg-success bg-opacity-10 text-center"><div className="fw-semibold text-success">Legit</div><div className="fs-4 text-success">{analytics.legit}</div></div></Col>
                    </Row>

                    <hr />
                    <div className="small">
                      <div className="d-flex justify-content-between"><span>Avg. confidence</span><span className="fw-semibold">{analytics.avgConfidence.toFixed(2)}</span></div>
                      <ProgressBar now={Math.min(100, analytics.avgConfidence * 100)} className="mt-1" />
                    </div>
                  </Card.Body>
                </Card>

                <Card className="mb-3 shadow-sm">
                  <Card.Body>
                    <Card.Title className="mb-3">Distribution</Card.Title>
                    <div className="mb-3">
                      <div className="d-flex justify-content-between small mb-1"><span className="text-danger">Phish</span><span className="fw-semibold">{analytics.phishPct}%</span></div>
                      <ProgressBar now={analytics.phishPct} variant="danger" />
                    </div>
                    <div className="mb-3">
                      <div className="d-flex justify-content-between small mb-1"><span className="text-success">Legit</span><span className="fw-semibold">{analytics.legitPct}%</span></div>
                      <ProgressBar now={analytics.legitPct} variant="success" />
                    </div>
                    <div>
                      <div className="d-flex justify-content-between small mb-1"><span className="text-primary">Clones</span><span className="fw-semibold">{analytics.clonePct}%</span></div>
                      <ProgressBar now={analytics.clonePct} variant="info" />
                    </div>
                  </Card.Body>
                </Card>

                <Card className="shadow-sm">
                  <Card.Body>
                    <Card.Title className="mb-3">Top Phishing Indicators</Card.Title>
                    {analytics.topIndicators.length === 0 ? (
                      <div className="text-muted small">No signals yet.</div>
                    ) : (
                      <ListGroup variant="flush">
                        {analytics.topIndicators.map(([name, count], i) => (
                          <ListGroup.Item key={i} className="d-flex justify-content-between">
                            <span>{name.replace(/_/g, " ")}</span>
                            <Badge bg="danger">{count}</Badge>
                          </ListGroup.Item>
                        ))}
                      </ListGroup>
                    )}
                  </Card.Body>
                </Card>
              </div>
            </Col>
          </Row>
        )}

        <style>{`
          .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 24px rgba(0,0,0,.12) !important;
          }
        `}</style>
      </Container>
    </div>
  );
}
