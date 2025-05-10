import React, { useState } from 'react';
import { useVideoContext } from '../context/VideoContext';
import './FrameMarker.css';

const FrameMarker = () => {
  const {
    videoData,
    markedFrames,
    currentFrame,
    setCurrentFrame,
    removeMarkedFrame,
    clearAllMarkedFrames
  } = useVideoContext();
  const [isExpanded, setIsExpanded] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [exportName, setExportName] = useState('');
  const [showExportDialog, setShowExportDialog] = useState(false);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  const jumpToFrame = (frame) => {
    setCurrentFrame(frame);
  };

  const handleRemoveFrame = (e, frameId) => {
    e.stopPropagation();
    removeMarkedFrame(frameId);
  };

  const handleRemoveAllFrames = () => {
    if (markedFrames.length === 0) return;

    if (window.confirm("Are you sure you want to remove all marked frames? This action cannot be undone.")) {
      clearAllMarkedFrames();
    }
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const filteredFrames = markedFrames.filter((mark) =>
    mark.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Get the most extreme min/max values for each metric across all filtered frames
  const aggregateFrameAnalytics = () => {
    if (!videoData || !videoData.analytics || filteredFrames.length === 0) {
      return null;
    }

    // Get frame numbers from filtered frames
    const frameNumbers = filteredFrames.map(mark => mark.frame);

    // Get analytics data for those frames
    const frameAnalytics = videoData.analytics.filter(data =>
      frameNumbers.includes(data.frame)
    );

    if (frameAnalytics.length === 0) {
      return null;
    }

    // Initialize metrics object to track min/max/total values
    const metrics = {};

    // First pass: collect all available metrics and initialize their values
    frameAnalytics.forEach(frameData => {
      if (!frameData.measurements) return;

      // Process lunge_distance if available
      if (frameData.measurements.lunge_distance !== undefined) {
        const value = frameData.measurements.lunge_distance;
        if (!metrics['lunge_distance']) {
          metrics['lunge_distance'] = {
            id: 'lunge_distance',
            values: [],
            min: value,
            max: value
          };
        }
        metrics['lunge_distance'].values.push(value);
        metrics['lunge_distance'].min = Math.min(metrics['lunge_distance'].min, value);
        metrics['lunge_distance'].max = Math.max(metrics['lunge_distance'].max, value);
      }

      // Process all other measurement sections
      Object.entries(frameData.measurements).forEach(([sectionKey, sectionData]) => {
        // Skip lunge_distance as it's already handled
        if (sectionKey === 'lunge_distance') return;

        // Process each metric in the section
        if (typeof sectionData === 'object') {
          Object.entries(sectionData).forEach(([metricKey, value]) => {
            if (value !== undefined && value !== null && !isNaN(value)) {
              const metricId = `${sectionKey}.${metricKey}`;

              if (!metrics[metricId]) {
                metrics[metricId] = {
                  id: metricId,
                  values: [],
                  min: value,
                  max: value
                };
              }

              metrics[metricId].values.push(value);
              metrics[metricId].min = Math.min(metrics[metricId].min, value);
              metrics[metricId].max = Math.max(metrics[metricId].max, value);
            }
          });
        }
      });
    });

    // Calculate average (current) value for each metric
    Object.values(metrics).forEach(metric => {
      const sum = metric.values.reduce((acc, val) => acc + val, 0);
      metric.avg = sum / metric.values.length;
    });

    return metrics;
  };

  // Create shot detection compatible export
  const createExportData = () => {
    const metricsData = aggregateFrameAnalytics();
    if (!metricsData) {
      alert("No analytics data available for the filtered frames.");
      return null;
    }

    const name = exportName.trim() ||
                 (searchTerm ? `${searchTerm} Filter` : 'Marked Frames Filter');

    // Format the export data to match ShotDetection import format
    const exportData = {
      version: "1.0",
      name: name,
      globalMargin: 10, // Default value
      frameWindowSize: 30, // Default value
      metrics: {}
    };

    // Convert our metrics to the expected format
    Object.values(metricsData).forEach(metric => {
      exportData.metrics[metric.id] = {
        selected: true, // Default to selected
        min: metric.min,
        current: metric.avg,
        max: metric.max
      };
    });

    return exportData;
  };

  // Export the aggregated analytics as a JSON file
  const handleExport = () => {
    const exportData = createExportData();
    if (!exportData) return;

    // Convert to JSON and create download link
    const jsonString = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    // Create a temporary link and trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = `${exportData.name.replace(/\s+/g, '_')}_filter.json`;
    document.body.appendChild(a);
    a.click();

    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Close the export dialog
    setShowExportDialog(false);
    setExportName('');
  };

  // Show export dialog
  const openExportDialog = () => {
    if (filteredFrames.length === 0) {
      alert("No frames to export. Please mark some frames first.");
      return;
    }

    setExportName(searchTerm ? `${searchTerm} Filter` : 'Marked Frames Filter');
    setShowExportDialog(true);
  };

  return (
    <div className="frame-marker">
      <div className="panel-header">
        <h3>Marked Frames</h3>
        <button className="toggle-button" onClick={toggleExpand}>
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {isExpanded && (
        <div className="panel-content">
          {!videoData || markedFrames.length === 0 ? (
            <div>
              <p>No frames marked yet</p>
              <p>Use the "Mark Frame" button to save important positions</p>
            </div>
          ) : (
            <>
              <div className="search-container">
                <input
                  type="text"
                  placeholder="Search frames..."
                  value={searchTerm}
                  onChange={handleSearchChange}
                  className="search-input"
                />
              </div>
              <div className="marked-frames-header">
                <span>{filteredFrames.length} of {markedFrames.length} marked frame(s)</span>
                <div className="header-buttons">
                  <button
                    className="export-button"
                    onClick={openExportDialog}
                  >
                    Export to Filter
                  </button>
                  <button
                    className="remove-all-button"
                    onClick={handleRemoveAllFrames}
                  >
                    Remove All
                  </button>
                </div>
              </div>
              <div className="marked-frames-list">
                {filteredFrames.map((mark) => (
                  <div
                    key={mark.id}
                    className={`marked-frame-item ${mark.frame === currentFrame ? 'active' : ''}`}
                    onClick={() => jumpToFrame(mark.frame)}
                  >
                    <div className="marked-frame-info">
                      <span className="marked-frame-name">{mark.name}</span>
                      <span className="marked-frame-number">Frame: {mark.frame}</span>
                    </div>
                    <button
                      className="remove-frame-button"
                      onClick={(e) => handleRemoveFrame(e, mark.id)}
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
              {searchTerm && filteredFrames.length === 0 && (
                <div className="no-results">
                  <p>No frames match your search</p>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Export Dialog */}
      {showExportDialog && (
        <div className="export-dialog-overlay">
          <div className="export-dialog">
            <h3>Export Frames to Filter</h3>
            <p>
              This will create a filter based on analytics from {filteredFrames.length} frame(s).
              The filter will contain min/max ranges for all metrics across these frames.
            </p>
            <div className="export-name-container">
              <label htmlFor="export-name">Filter Name:</label>
              <input
                id="export-name"
                type="text"
                value={exportName}
                onChange={(e) => setExportName(e.target.value)}
                placeholder="Enter a name for this filter"
                className="export-name-input"
              />
            </div>
            <div className="export-dialog-buttons">
              <button
                className="cancel-button"
                onClick={() => setShowExportDialog(false)}
              >
                Cancel
              </button>
              <button
                className="confirm-export-button"
                onClick={handleExport}
              >
                Export
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FrameMarker;