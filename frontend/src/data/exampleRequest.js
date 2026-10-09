export const exampleRequest = {
  observation: { agent_location: 'room_a', door: { location: 'room_b', locked: true }, key: { location: 'box', available: true }, box: { location: 'room_a', open: false } },
  goal: { description: 'Masuk ke room_b', target_state: { agent_location: 'room_b' } },
  state: { agent_location: 'room_a', has_key: false, door_unlocked: false },
  world_model: { actions: ['open_box','take_key','unlock_door','open_door','move_to_room_b'], constraints: ['Pintu harus dibuka sebelum masuk room_b','Pintu terkunci harus dibuka menggunakan kunci'] },
  relevant_memory: [{ type: 'semantic', content: 'Kunci dapat digunakan untuk membuka pintu yang terkunci' }, { type: 'episodic', content: 'Untuk mengambil kunci di dalam kotak, kotak harus dibuka terlebih dahulu' }],
  strategy: { approach: 'goal_directed_planning', prefer_shortest_plan: true },
  self_awareness: { capabilities: ['open_box','take_key','unlock_door','open_door','move_to_room_b'], limitations: ['Tidak dapat membuka pintu terkunci tanpa kunci'] },
  calibration: { confidence_threshold: 0.7 }, metacognition: { enabled: true, reflection_mode: 'post_episode' },
  metadata: { source: 'frontend_dashboard', experiment: 'metacognition_integration', version: 1 }, include_trace: true, include_metacognition: true,
  episode: { episode_id: 'episode-ui-001', goal: 'Masuk ke room_b', action: 'open_box', success: true, progress_before: 0, progress_after: 0.2, progress_delta: 0.2, observations: ['Agent berada di room_a','Kunci berada di dalam kotak'] },
  deterministic_reflection: { episode_id: 'episode-ui-001', goal: 'Masuk ke room_b', action: 'open_box', success: true, progress_before: 0, progress_after: 0.2, progress_delta: 0.2, diagnosis: 'Kemajuan awal berhasil karena kotak berhasil dibuka', lesson: 'Kotak perlu dibuka sebelum kunci dapat diambil', next_step: 'Ambil kunci dari kotak', evidence_quality: 0.9, confidence: 0.9, change_required: false }
};
