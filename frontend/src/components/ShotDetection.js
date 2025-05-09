import React, { useState, useEffect } from 'react';
import { useVideoContext } from '../context/VideoContext';
import './ShotDetection.css';

const ShotDetection = () => {
  const { videoData, currentFrame, markCustomFrames } = useVideoContext();
  const [isExpanded, setIsExpanded] = useState(true);
  const [eventName, setEventName] = useState('');
  const [selectedMetrics, setSelectedMetrics] = useState({});
  const [metricValues, setMetricValues] = useState({});
  const [globalMargin, setGlobalMargin] = useState(10); // Default 10% margin
  const [frameWindowSize, setFrameWindowSize] = useState(30); // Default window size of 30 frames
  const [availableMetrics, setAvailableMetrics] = useState([]);

  // Initialize available metrics when currentFrame changes and has data
  useEffect(() => {
    if (!videoData || !videoData.analytics || currentFrame === null) return;

    const frameAnalytics = videoData.analytics.find(data => data.frame === currentFrame);
    if (!frameAnalytics || !frameAnalytics.measurements) return;

    // Collect all available metrics from all sections
    const metrics = [];

    Object.entries(frameAnalytics.measurements).forEach(([sectionKey, sectionData]) => {
      if (sectionKey === 'lunge_distance') {
        metrics.push({
          id: 'lunge_distance',
          label: 'Lunge Distance',
          value: frameAnalytics.measurements.lunge_distance,
          section: 'Lunge Analysis'
        });
        return;
      }

      Object.entries(sectionData).forEach(([key, value]) => {
        if (value !== undefined && value !== null && !isNaN(value)) {
          metrics.push({
            id: `${sectionKey}.${key}`,
            label: formatMetricLabel(key),
            value: value,
            section: formatSectionTitle(sectionKey)
          });
        }
      });
    });

    setAvailableMetrics(metrics);

    // Initialize selection state and values with min/max range
    const initialSelectedState = {};
    const initialValues = {};

    metrics.forEach(metric => {
      initialSelectedState[metric.id] = false;
      const currentValue = metric.value;
      const marginAmount = currentValue * (globalMargin / 100);

      initialValues[metric.id] = {
        current: currentValue,
        min: currentValue - marginAmount,
        max: currentValue + marginAmount
      };
    });

    setSelectedMetrics(initialSelectedState);
    setMetricValues(initialValues);
  }, [videoData, currentFrame, globalMargin]);

  // Update min/max values when global margin changes
  useEffect(() => {
    if (availableMetrics.length === 0) return;

    setMetricValues(prev => {
      const updated = { ...prev };

      Object.keys(updated).forEach(metricId => {
        const currentValue = updated[metricId].current;
        const marginAmount = currentValue * (globalMargin / 100);

        updated[metricId] = {
          ...updated[metricId],
          min: currentValue - marginAmount,
          max: currentValue + marginAmount
        };
      });

      return updated;
    });
  }, [globalMargin, availableMetrics]);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  // Format section titles to be more readable
  const formatSectionTitle = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Format metric labels to be more readable
  const formatMetricLabel = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const handleMetricToggle = (metricId) => {
    setSelectedMetrics(prev => ({
      ...prev,
      [metricId]: !prev[metricId]
    }));
  };

  const handleValueChange = (metricId, field, value) => {
    setMetricValues(prev => {
      const updated = { ...prev };
      updated[metricId] = { ...updated[metricId], [field]: parseFloat(value) };

      // If we're updating the current value, also update min/max based on margin
      if (field === 'current') {
        const marginAmount = parseFloat(value) * (globalMargin / 100);
        updated[metricId].min = parseFloat(value) - marginAmount;
        updated[metricId].max = parseFloat(value) + marginAmount;
      }

      return updated;
    });
  };

  const handleGlobalMarginChange = (value) => {
    setGlobalMargin(parseFloat(value));
  };

  const handleFrameWindowSizeChange = (value) => {
    setFrameWindowSize(parseInt(value, 10));
  };

  // Function to check if all selected metrics appear anywhere within a window of frames
  const checkMetricsInFrameWindow = (startFrame, endFrame, filters) => {
    if (!videoData || !videoData.analytics) return false;

    // Get all the frames in this window
    const framesInWindow = videoData.analytics.filter(
      frameData => frameData.frame >= startFrame && frameData.frame <= endFrame
    );

    if (framesInWindow.length === 0) return false;

    // Check if each filter is satisfied by at least one frame in the window
    const satisfiedFilters = filters.map(filter => {
      return framesInWindow.some(frameData => {
        let actualValue;

        // Extract the actual value from the frame data
        if (filter.id === 'lunge_distance') {
          actualValue = frameData.measurements.lunge_distance;
        } else {
          const [section, metric] = filter.id.split('.');
          if (frameData.measurements[section] &&
              frameData.measurements[section][metric] !== undefined) {
            actualValue = frameData.measurements[section][metric];
          }
        }

        // Check if value is within the min-max range
        return actualValue !== undefined &&
               !isNaN(actualValue) &&
               actualValue >= filter.min &&
               actualValue <= filter.max;
      });
    });

    // All filters must be satisfied by at least one frame in the window
    return satisfiedFilters.every(isSatisfied => isSatisfied);
  };

  // Function to consolidate frames that are within 100 frames of each other
  const consolidateFrames = (frames) => {
    if (!frames || frames.length <= 1) return frames;

    // Sort frames numerically
    const sortedFrames = [...frames].sort((a, b) => a - b);

    // Group frames that are within 100 frames of each other
    const groups = [];
    let currentGroup = [sortedFrames[0]];

    for (let i = 1; i < sortedFrames.length; i++) {
      const currentFrame = sortedFrames[i];
      const lastFrameInGroup = currentGroup[currentGroup.length - 1];

      if (currentFrame - lastFrameInGroup <= 100) {
        // Frame is within range, add to current group
        currentGroup.push(currentFrame);
      } else {
        // Frame is outside range, finish current group and start a new one
        groups.push(currentGroup);
        currentGroup = [currentFrame];
      }
    }

    // Add the last group if it has elements
    if (currentGroup.length > 0) {
      groups.push(currentGroup);
    }

    // Calculate median frame for each group
    const medianFrames = groups.map(group => {
      // Sort the group (should already be sorted but just to be sure)
      group.sort((a, b) => a - b);

      // Find median
      const mid = Math.floor(group.length / 2);
      if (group.length % 2 === 0) {
        // Even number of elements, average the middle two
        return Math.round((group[mid - 1] + group[mid]) / 2);
      } else {
        // Odd number of elements, return the middle one
        return group[mid];
      }
    });

    return medianFrames;
  };

  const handleDetectFrames = () => {
    if (!eventName.trim()) {
      alert("Please enter an event name");
      return;
    }

    // Check if any metrics are selected
    const anySelected = Object.values(selectedMetrics).some(selected => selected);
    if (!anySelected) {
      alert("Please select at least one metric for detection");
      return;
    }

    // Validate frame window size
    if (frameWindowSize <= 0) {
      alert("Frame window size must be a positive number");
      return;
    }

    // Get selected metrics with their min/max values
    const filters = Object.entries(selectedMetrics)
      .filter(([_, selected]) => selected)
      .map(([metricId]) => {
        return {
          id: metricId,
          min: metricValues[metricId].min,
          max: metricValues[metricId].max
        };
      });

    // Find matching frame windows
    const matchingWindowStarts = [];

    if (videoData && videoData.analytics) {
      // Get all frame numbers and sort them
      const allFrames = videoData.analytics.map(data => data.frame).sort((a, b) => a - b);

      // Check each possible window start position
      for (let i = 0; i < allFrames.length; i++) {
        const startFrame = allFrames[i];
        const endFrame = startFrame + frameWindowSize - 1;

        // Skip if the end frame would be beyond the last available frame
        if (endFrame > allFrames[allFrames.length - 1]) break;

        // Check if all selected metrics appear somewhere in this frame window
        if (checkMetricsInFrameWindow(startFrame, endFrame, filters)) {
          matchingWindowStarts.push(startFrame);
        }
      }
    }

    // Consolidate frame windows that are close to each other
    const consolidatedFrames = consolidateFrames(matchingWindowStarts);

    // Add the matching frames to marked frames
    if (consolidatedFrames.length > 0) {
      markCustomFrames(eventName, consolidatedFrames);

      // Prepare message with original and consolidated counts
      const message = matchingWindowStarts.length === consolidatedFrames.length
        ? `Found ${consolidatedFrames.length} matching frame sequences for "${eventName}"`
        : `Found ${matchingWindowStarts.length} matching frame sequences, consolidated to ${consolidatedFrames.length} for "${eventName}"`;

      alert(message);
    } else {
      alert("No matching frame sequences found. Try adjusting your criteria.");
    }
  };

  // Group metrics by section
  const metricsBySection = availableMetrics.reduce((acc, metric) => {
    if (!acc[metric.section]) {
      acc[metric.section] = [];
    }
    acc[metric.section].push(metric);
    return acc;
  }, {});

  return (
    <div className="frame-marker shot-detection">
      <div className="panel-header">
        <h3>Shot Detection</h3>
        <button className="toggle-button" onClick={toggleExpand}>
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {isExpanded && (
        <div className="panel-content">
          {!videoData ? (
            <div>
              <p>No video data loaded</p>
              <p>Load a video to use shot detection features</p>
            </div>
          ) : (
            <>
              <div className="event-name-container">
                <label>Event Name:</label>
                <input
                  type="text"
                  value={eventName}
                  onChange={(e) => setEventName(e.target.value)}
                  placeholder="e.g., Perfect Lunge"
                  className="event-name-input"
                />
              </div>

              <div className="global-controls">
                <div className="frame-window-control">
                  <label>Frame Window Size:</label>
                  <input
                    type="text"
                    value={frameWindowSize}
                    onChange={(e) => handleFrameWindowSizeChange(e.target.value)}
                    min="1"
                    className="frame-window-input"
                  />
                  <span className="input-info">frames</span>
                </div>
                <div className="global-margin-control">
                  <label>Global Margin:</label>
                  <input
                    type="text"
                    value={globalMargin}
                    onChange={(e) => handleGlobalMarginChange(e.target.value)}
                    min="0"
                    max="1000"
                    className="global-margin-input"
                  />
                  <span className="input-info">%</span>
                </div>
              </div>

              <div className="detection-info">
                <p className="detection-explanation">
                  Detection will find frame sequences of {frameWindowSize} frames where all selected metrics
                  appear within their min/max ranges (in any order within the sequence).
                </p>
              </div>

              <div className="metrics-header">
                <span>{Object.values(selectedMetrics).filter(selected => selected).length} metric(s) selected</span>
                <button
                  className="detect-button"
                  onClick={handleDetectFrames}
                >
                  Find Matching Sequences
                </button>
              </div>

              <div className="metrics-list">
                {Object.entries(metricsBySection).map(([section, metrics]) => (
                  <div key={section} className="metrics-section">
                    <h4 className="section-title">{section}</h4>
                    {metrics.map(metric => (
                      <div key={metric.id} className="metric-item">
                        <div className="metric-header">
                          <input
                            type="checkbox"
                            checked={selectedMetrics[metric.id] || false}
                            onChange={() => handleMetricToggle(metric.id)}
                            id={`metric-${metric.id}`}
                          />
                          <label
                            htmlFor={`metric-${metric.id}`}
                            className="metric-label"
                          >
                            {metric.label}
                          </label>
                        </div>
                        <div className="metric-controls">
                          <div className="range-values">
                            <div className="min-value">
                              <label>Min:</label>
                              <input
                                type="text"
                                value={metricValues[metric.id]?.min.toFixed(2) || 0}
                                onChange={(e) => handleValueChange(metric.id, 'min', e.target.value)}
                                disabled={!selectedMetrics[metric.id]}
                                
                              />
                            </div>
                            <div className="current-value">
                              <label>Current:</label>
                              <input
                                type="text"
                                value={metricValues[metric.id]?.current.toFixed(2) || 0}
                                onChange={(e) => handleValueChange(metric.id, 'current', e.target.value)}
                                disabled={!selectedMetrics[metric.id]}
                                
                              />
                            </div>
                            <div className="max-value">
                              <label>Max:</label>
                              <input
                                type="text"
                                value={metricValues[metric.id]?.max.toFixed(2) || 0}
                                onChange={(e) => handleValueChange(metric.id, 'max', e.target.value)}
                                disabled={!selectedMetrics[metric.id]}
                                
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ShotDetection;