import { useState, useEffect } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Users, LogIn, LogOut, Video, StopCircle, RefreshCw, Eye, EyeOff, Clock, List, Download, FileText, Film } from 'lucide-react'
import './App.css'

const API_BASE = 'http://localhost:8001'

function App() {
  const [stats, setStats] = useState({ entry_count: 0, exit_count: 0, unique_visitors: 0 })
  const [faces, setFaces] = useState([])
  const [events, setEvents] = useState([])
  const [logFiles, setLogFiles] = useState([])
  const [sessions, setSessions] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [trackerRunning, setTrackerRunning] = useState(false)
  const [inputType, setInputType] = useState('rtsp')
  const [rtspUrl, setRtspUrl] = useState('')
  const [videoFile, setVideoFile] = useState(null)
  const [showViz, setShowViz] = useState(true)
  const [activeTab, setActiveTab] = useState('visitors')

  const fetchData = async () => {
    try {
      const [
        { data: statData }, 
        { data: faceData }, 
        { data: eventData }, 
        { data: sessionData },
        { data: statusData }
      ] = await Promise.all([
        axios.get(`${API_BASE}/api/dashboard/stats`),
        axios.get(`${API_BASE}/api/faces`),
        axios.get(`${API_BASE}/api/events?limit=100`),
        axios.get(`${API_BASE}/api/sessions`),
        axios.get(`${API_BASE}/api/tracker/status`)
      ])
      
      if (!statData.error) setStats(statData)
      if (!faceData.error) setFaces(faceData)
      if (!eventData.error && Array.isArray(eventData)) setEvents(eventData)
      if (!sessionData.error && Array.isArray(sessionData)) setSessions(sessionData)
      if (statusData && typeof statusData.running === 'boolean') setTrackerRunning(statusData.running)
    } catch (error) {
      console.error("Error fetching data:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const fetchLogFiles = async () => {
    try {
      const { data } = await axios.get(`${API_BASE}/api/logs`)
      if (Array.isArray(data)) setLogFiles(data)
    } catch (err) {
      console.error('Error fetching log files:', err)
    }
  }

  useEffect(() => {
    fetchData()
    fetchLogFiles()
    const interval = setInterval(fetchData, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleStart = async (e) => {
    e.preventDefault()
    setTrackerRunning(true)
    const formData = new FormData()
    formData.append("source_type", inputType)
    formData.append("show_visualization", showViz ? "true" : "false")
    if (inputType === 'rtsp') {
      formData.append("rtsp_url", rtspUrl)
    } else if (videoFile) {
      formData.append("video_file", videoFile)
    }

    try {
      await axios.post(`${API_BASE}/api/tracker/start`, formData)
      alert("Tracker started processing!")
    } catch (err) {
      console.error(err)
      alert("Failed to start tracker")
      setTrackerRunning(false)
    }
  }

  const handleStop = async () => {
    try {
      await axios.post(`${API_BASE}/api/tracker/stop`)
      setTrackerRunning(false)
      alert("Sent stop signal")
    } catch (err) {
      console.error(err)
    }
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <h1>Intelligent Face Tracker</h1>
        <div className="controls">
          <select 
            className="input-field" 
            style={{ minWidth: '120px' }}
            value={inputType} 
            onChange={e => setInputType(e.target.value)}
          >
            <option value="rtsp">RTSP Stream</option>
            <option value="upload">Upload Video</option>
          </select>
          
          {inputType === 'rtsp' ? (
            <input 
              className="input-field"
              placeholder="rtsp://admin:password@ip..." 
              value={rtspUrl}
              onChange={e => setRtspUrl(e.target.value)}
            />
          ) : (
            <input 
              className="input-field"
              type="file" 
              accept="video/mp4"
              onChange={e => setVideoFile(e.target.files[0])}
            />
          )}

          {/* Visualization toggle */}
          <button
            className={`btn ${showViz ? 'btn-viz-on' : ''}`}
            onClick={() => setShowViz(!showViz)}
            title={showViz ? "Live visualization ON" : "Live visualization OFF"}
          >
            {showViz ? <Eye size={18} /> : <EyeOff size={18} />}
            <span className="viz-label">{showViz ? 'Viz ON' : 'Viz OFF'}</span>
          </button>

          {!trackerRunning ? (
            <button className="btn btn-primary" onClick={handleStart}>
              <Video size={18} /> Start Tracking
            </button>
          ) : (
            <button className="btn btn-danger" onClick={handleStop}>
              <StopCircle size={18} /> Stop
            </button>
          )}
        </div>
      </header>

      {/* Stats Grid */}
      <div className="stats-grid">
        <div className="glass-panel stat-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-color)' }}>
            <Users size={20} />
            <span className="stat-label">Unique Visitors</span>
          </div>
          <span className="stat-value">{stats.unique_visitors}</span>
        </div>
        
        <div className="glass-panel stat-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--success)' }}>
            <LogIn size={20} />
            <span className="stat-label">Total Entries</span>
          </div>
          <span className="stat-value">{stats.entry_count}</span>
        </div>

        <div className="glass-panel stat-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--danger)' }}>
            <LogOut size={20} />
            <span className="stat-label">Total Exits</span>
          </div>
          <span className="stat-value">{stats.exit_count}</span>
        </div>
        
        <div className="glass-panel stat-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexDirection: 'row' }}>
          <div>
            <span className="stat-label">System Status</span>
            <div style={{ fontWeight: 600, fontSize: '1.25rem', color: trackerRunning ? 'var(--success)' : 'var(--text-secondary)' }}>
              {trackerRunning ? 'Processing' : 'Standby'}
            </div>
          </div>
          <motion.div 
            animate={{ rotate: trackerRunning ? 360 : 0 }} 
            transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
          >
            <Activity color={trackerRunning ? 'var(--success)' : 'var(--text-secondary)'} size={32} />
          </motion.div>
        </div>
      </div>

      {/* Tab Switcher */}
      <div className="tab-bar">
        <button className={`tab-btn ${activeTab === 'visitors' ? 'active' : ''}`} onClick={() => setActiveTab('visitors')}>
          <Users size={16} /> Recent Visitors
        </button>
        <button className={`tab-btn ${activeTab === 'sessions' ? 'active' : ''}`} onClick={() => setActiveTab('sessions')}>
          <Film size={16} /> Sessions
        </button>
        <button className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`} onClick={() => setActiveTab('logs')}>
          <List size={16} /> Event Log
        </button>
        <button className={`tab-btn ${activeTab === 'downloads' ? 'active' : ''}`} onClick={() => { setActiveTab('downloads'); fetchLogFiles() }}>
          <Download size={16} /> Downloads
        </button>
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={fetchData} style={{ padding: '0.4rem 0.8rem', fontSize: '0.875rem' }}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* Visitors Tab */}
      {activeTab === 'visitors' && (
        <div className="visitors-grid">
          <AnimatePresence>
            {faces.map((face, index) => (
              <motion.div 
                key={face.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ delay: index * 0.05 }}
                className="glass-panel visitor-card"
              >
                <div className="visitor-header">
                  <div>
                    <div className="visitor-id">{face.id.split('-')[0]}***</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                      First seen: {new Date(face.first_seen).toLocaleTimeString()}
                    </div>
                  </div>
                  <div className="visit-badge">{face.visit_count} visit(s)</div>
                </div>
                
                <div className="visitor-images">
                  <div className="image-container">
                    <span className="image-label">ENTRY</span>
                    {face.latest_entry_image ? (
                      <img src={`${API_BASE}/images/${face.latest_entry_image}`} alt="Entry" />
                    ) : (
                      <div className="placeholder-img">No Image</div>
                    )}
                  </div>
                  
                  <div className="image-container">
                    <span className="image-label">EXIT</span>
                    {face.latest_exit_image ? (
                      <img src={`${API_BASE}/images/${face.latest_exit_image}`} alt="Exit" />
                    ) : (
                      <div className="placeholder-img">Still Present</div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {faces.length === 0 && !isLoading && (
            <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '3rem', color: 'var(--text-secondary)' }}>
              <Users size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
              <p>No visitors recorded yet.</p>
            </div>
          )}
        </div>
      )}

      {/* Event Log Tab */}
      {activeTab === 'logs' && (
        <div className="glass-panel event-log-panel">
          {events.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
              <Clock size={32} style={{ opacity: 0.5, marginBottom: '0.5rem' }} />
              <p>No events recorded yet.</p>
            </div>
          ) : (
            <table className="event-table">
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Face ID</th>
                  <th>Track ID</th>
                  <th>Frame</th>
                  <th>Time</th>
                  <th>Image</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {events.map((evt, i) => (
                    <motion.tr
                      key={evt.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.02 }}
                    >
                      <td>
                        <span className={`event-badge ${evt.event_type}`}>
                          {evt.event_type === 'entry' ? <LogIn size={12} /> : <LogOut size={12} />}
                          {evt.event_type.toUpperCase()}
                        </span>
                      </td>
                      <td className="mono">{evt.face_id?.split('-')[0]}***</td>
                      <td className="mono">T:{evt.track_id}</td>
                      <td className="mono">#{evt.frame_number}</td>
                      <td>{new Date(evt.occurred_at).toLocaleTimeString()}</td>
                      <td>
                        {evt.image_path ? (
                          <img 
                            src={`${API_BASE}/images/${evt.image_path}`}
                            alt={evt.event_type}
                            className="event-thumb"
                          />
                        ) : '—'}
                      </td>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Sessions Tab */}
      {activeTab === 'sessions' && (
        <div className="glass-panel event-log-panel">
          {sessions.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
              <Film size={32} style={{ opacity: 0.4, marginBottom: '0.5rem' }} />
              <p>No sessions recorded yet.</p>
            </div>
          ) : (
            <table className="event-table">
              <thead>
                <tr>
                  <th>Video</th>
                  <th>Unique Visitors</th>
                  <th>Entries</th>
                  <th>Exits</th>
                  <th>Duration</th>
                  <th>Run Time</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((s) => {
                  const fmtDuration = (sec) => {
                    if (!sec) return '—'
                    const m = Math.floor(sec / 60)
                    const ss = Math.floor(sec % 60)
                    return `${m}m ${ss}s`
                  }
                  return (
                    <motion.tr
                      key={s.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <Film size={14} color="var(--accent-color)" />
                          <span style={{ fontWeight: 500 }}>{s.video_name}</span>
                        </div>
                      </td>
                      <td>
                        <span style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-color)' }}>
                          {s.unique_visitors}
                        </span>
                      </td>
                      <td>
                        <span className="event-badge entry"><LogIn size={12} /> {s.entry_count}</span>
                      </td>
                      <td>
                        <span className="event-badge exit"><LogOut size={12} /> {s.exit_count}</span>
                      </td>
                      <td className="mono">{fmtDuration(s.duration_s)}</td>
                      <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        {new Date(s.started_at).toLocaleString()}
                      </td>
                    </motion.tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Downloads Tab */}
      {activeTab === 'downloads' && (
        <div className="glass-panel" style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
            <Download size={20} color="var(--accent-color)" />
            <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>Raw Log Files</h3>
          </div>
          
          {logFiles.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
              <FileText size={32} style={{ opacity: 0.4, marginBottom: '0.5rem' }} />
              <p>No log files available.</p>
            </div>
          ) : (
            <div className="log-file-list">
              {logFiles.map((file) => (
                <div key={file.name} className="log-file-row">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <FileText size={18} color="var(--text-secondary)" />
                    <div>
                      <div style={{ fontWeight: 500 }}>{file.name}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        {(file.size_bytes / 1024).toFixed(1)} KB
                      </div>
                    </div>
                  </div>
                  <a
                    href={`${API_BASE}/api/logs/download/${file.name}`}
                    download={file.name}
                    className="btn btn-primary"
                    style={{ padding: '0.4rem 1rem', fontSize: '0.85rem', textDecoration: 'none' }}
                  >
                    <Download size={14} /> Download
                  </a>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
