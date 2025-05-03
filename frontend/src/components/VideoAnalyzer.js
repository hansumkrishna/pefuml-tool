import React, { useRef, useState, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useVideoContext } from '../context/VideoContext';
import AnalyticsPanel from './AnalyticsPanel';
import FrameMarker from './FrameMarker';
import './VideoAnalyzer.css';

const VideoAnalyzer = () => {
  const { 
    videoData, 
    displayOptions, 
    currentFrame, 
    setCurrentFrame,
    isPlaying, 
    setIsPlaying,
    markCurrentFrame
  } = useVideoContext();

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const threeContainerRef = useRef(null);
  
  const [scene, setScene] = useState(null);
  const [camera, setCamera] = useState(null);
  const [renderer, setRenderer] = useState(null);
  const [keypoints, setKeypoints] = useState(null);
  const [skeleton, setSkeleton] = useState(null);
  
  // Initialize Three.js scene
  useEffect(() => {
    if (!threeContainerRef.current || !displayOptions.showKeypoints) return;
    
    // Create scene
    const newScene = new THREE.Scene();
    newScene.background = new THREE.Color(0x222222); // Dark gray background
    newScene.background.alpha = 1;
    
    // Create camera
    const newCamera = new THREE.PerspectiveCamera(
      75, 
      threeContainerRef.current.clientWidth / threeContainerRef.current.clientHeight, 
      0.1, 
      1000
    );
    newCamera.position.z = 2;
    
    // Create renderer with alpha: true for transparency
    const newRenderer = new THREE.WebGLRenderer({ 
      alpha: true,
      antialias: true 
    });
    newRenderer.setSize(
      threeContainerRef.current.clientWidth, 
      threeContainerRef.current.clientHeight
    );
    // Set clear color with 0 alpha (fully transparent)
    newRenderer.setClearColor(0x000000, 0);
    
    // Clear container and append renderer
    threeContainerRef.current.innerHTML = '';
    threeContainerRef.current.appendChild(newRenderer.domElement);
    
    // Add orbit controls
    const controls = new OrbitControls(newCamera, newRenderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    newScene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    newScene.add(directionalLight);
    
    // Create skeleton group
    const newSkeleton = new THREE.Group();
    newScene.add(newSkeleton);
    
    setScene(newScene);
    setCamera(newCamera);
    setRenderer(newRenderer);
    setSkeleton(newSkeleton);
    
    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      newRenderer.render(newScene, newCamera);
    };
    
    animate();

    // Resize handler
    const handleResize = () => {
      if (!threeContainerRef.current) return;

      const containerWidth = threeContainerRef.current.clientWidth;
      const containerHeight = threeContainerRef.current.clientHeight;

      newCamera.aspect = containerWidth / containerHeight;
      newCamera.updateProjectionMatrix();

      newRenderer.setSize(containerWidth, containerHeight);
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      
      if (newRenderer && threeContainerRef.current) {
        threeContainerRef.current.removeChild(newRenderer.domElement);
      }
    };
  }, [displayOptions.showKeypoints]);
  useEffect(() => {
      console.log("Display options:", displayOptions);
      console.log("Current frame:", currentFrame);
      if (videoData) {
        console.log("Frame data available:", !!videoData.keypoints_timeline[currentFrame]);
      }
    }, [displayOptions, currentFrame, videoData]);

  // Update skeleton based on current frame's keypoints
  useEffect(() => {
    if (!videoData || !skeleton || !scene || currentFrame === null) return;
    
    const frameData = videoData.keypoints_timeline[currentFrame];
    if (!frameData || !frameData.keypoints) return;
    
    setKeypoints(frameData.keypoints);
    
    // Clear existing skeleton
    while (skeleton.children.length > 0) {
      skeleton.remove(skeleton.children[0]);
    }
    
    // Create transparent materials for joints
    const jointMaterial = new THREE.MeshPhongMaterial({
      color: 0xff0000,
      transparent: true,
      opacity: 1 // Adjust this value between 0-1 for desired transparency
    });

    // Create transparent materials for bones/lines
    const boneMaterial = new THREE.LineBasicMaterial({
      color: 0x00ff00,
      transparent: true,
      opacity: 1 // Adjust this value between 0-1 for desired transparency
    });

    frameData.keypoints.forEach((keypoint, index) => {
      if (keypoint.visibility > 0) {
        const jointGeometry = new THREE.SphereGeometry(0.02, 16, 16);
        const joint = new THREE.Mesh(jointGeometry, jointMaterial);
        
        // Scale and position
        joint.position.set(keypoint.x, -keypoint.y, -keypoint.z);
        skeleton.add(joint);
      }
    });
    
    // Define connections (based on MediaPipe pose connections)
    const connections = [
      // Torso
      [11, 12], [12, 24], [24, 23], [23, 11],
      // Right arm
      [12, 14], [14, 16], [16, 18], [18, 20], [16, 22],
      // Left arm
      [11, 13], [13, 15], [15, 17], [17, 19], [15, 21],
      // Right leg
      [24, 26], [26, 28], [28, 30], [28, 32],
      // Left leg
      [23, 25], [25, 27], [27, 29], [27, 31],
      // Face
      [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
      // Extra connections
      [9, 10], [11, 23], [12, 24],
    ];
    
    connections.forEach(([i, j]) => {
      const point1 = frameData.keypoints[i];
      const point2 = frameData.keypoints[j];
      
      if (point1 && point2 && point1.visibility > 0 && point2.visibility > 0) {
        const points = [
          new THREE.Vector3(point1.x, -point1.y, -point1.z),
          new THREE.Vector3(point2.x, -point2.y, -point2.z)
        ];

        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, boneMaterial);
        skeleton.add(line);
      }
    });
    
    // Center and scale skeleton
    skeleton.position.set(0, 0, 0);
    skeleton.scale.set(2, 2, 2);
    
  }, [videoData, skeleton, scene, currentFrame]);
  
  // Video player controls
  useEffect(() => {
    if (!videoRef.current || !videoData) return;
    
    const video = videoRef.current;
    
    const handleTimeUpdate = () => {
      if (!isPlaying) return;
      
      const fps = videoData.fps;
      const currentTime = video.currentTime;
      const frame = Math.floor(currentTime * fps);
      
      setCurrentFrame(frame);
    };
    
    video.addEventListener('timeupdate', handleTimeUpdate);
    
    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [videoData, isPlaying, setCurrentFrame]);
  
  // Play/pause video
  useEffect(() => {
    if (!videoRef.current) return;
    
    if (isPlaying) {
      videoRef.current.play().catch(error => {
        console.error("Error playing video:", error);
        setIsPlaying(false);
      });
    } else {
      videoRef.current.pause();
    }
  }, [isPlaying, setIsPlaying]);

  // Seek to specific frame
  useEffect(() => {
    if (!videoRef.current || !videoData || currentFrame === null) return;
    
    const fps = videoData.fps;
    const targetTime = currentFrame / fps;
    
    if (Math.abs(videoRef.current.currentTime - targetTime) > 0.1) {
      videoRef.current.currentTime = targetTime;
    }
  }, [videoData, currentFrame]);
  
  const handlePlayPause = () => {
      setIsPlaying(!isPlaying);
      console.log("Attempting to:", !isPlaying ? "play" : "pause");
      // Try to force play directly
      if (!isPlaying) {
        videoRef.current.play().catch(e => console.error("Play error:", e));
      }
    };
  
  const handleSeek = (e) => {
    if (!videoData) return;

    const totalFrames = videoData.total_frames;
    const seekPosition = parseInt(e.target.value);

    setCurrentFrame(seekPosition);
  };
  
  const handleMarkFrame = () => {
    if (currentFrame === null) return;
    
    const frameName = prompt("Enter a name for this marked frame:");
    if (frameName) {
      markCurrentFrame(frameName);
    }
  };
  
  if (!videoData) {
    return (
      <div className="video-analyzer empty-state">
        <h2>PEFUML Tool</h2>
        <p>Upload a video from the sidebar to get started</p>
      </div>
    );
  }
  
  return (
    <div className="video-analyzer">
      <div className="side-by-side-container">
        {displayOptions.showVideo && (
          <div className="video-wrapper">
            <video 
              ref={videoRef}
              className="video-player"
              src={`http://localhost:5000/api/uploads/${videoData.id}_${videoData.filename}`}
              controls={false}
            />
          </div>
        )}
        
        {displayOptions.showKeypoints && (
          <div 
            ref={threeContainerRef}
            className="keypoints-container"
          />
        )}
      </div>
      
      <div className="video-controls">
        <button onClick={handlePlayPause} className="control-button">
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        
        <input
          type="range"
          min="0"
          max={videoData.total_frames - 1}
          value={currentFrame || 0}
          onChange={handleSeek}
          className="seek-slider"
        />
        
        <span className="frame-counter">
          Frame: {currentFrame} / {videoData.total_frames - 1}
        </span>
        
        <button onClick={handleMarkFrame} className="mark-button">
          Mark Frame
        </button>
      </div>
      
      <div className="panels-container">
        <AnalyticsPanel />
        <FrameMarker />
      </div>
    </div>
      

      
      
  );
};

export default VideoAnalyzer;