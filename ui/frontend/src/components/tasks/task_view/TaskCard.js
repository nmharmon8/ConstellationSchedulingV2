import React from 'react';

const TaskCard = ({ task, isCompleted = false, onClick }) => {
  if (!task) return null;

  const truncateId = (id) => {
    return id?.toString().slice(0, 12) + (id?.toString().length > 12 ? '...' : '');
  };

  const getTaskTypeBadges = () => {
    const badges = [];
    if (task.is_data_downlink) badges.push('Downlink');
    if (task.is_access_task) badges.push('Access');
    if (task.is_noop) badges.push('No-Op');
    if (task.is_charge) badges.push('Charge');
    if (task.is_collection) badges.push('Collection');
    if (task.is_desat) badges.push('Desat');
    return badges;
  };

  return (
    <div
      key={task.id}
      className="group relative bg-gradient-to-br from-space-darker to-space-dark border border-gray-600/50 rounded-lg p-4 mb-3 mr-1 
      hover:border-space-cyan/50 hover:shadow-lg hover:shadow-space-cyan/20 transition-all duration-300 cursor-pointer
      hover:scale-[1.01] hover:-translate-y-0.5
      before:absolute before:inset-0 before:border-l-2 before:border-space-cyan/30 before:rounded-lg
      after:absolute after:inset-0 after:bg-space-cyan/5 after:opacity-0 after:rounded-lg after:transition-opacity hover:after:opacity-100"
      onClick={() => onClick(task)}
    >
      {/* Header Section */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-space-cyan font-mono text-lg font-bold tracking-wider 
        bg-space-darker/60 px-2 py-0.5 rounded">
          {truncateId(task.id)}
        </span>
        <span className="px-3 py-1 text-sm rounded-full bg-space-dark/80 text-space-cyan border border-space-cyan/30 
        backdrop-blur-sm shadow-inner group-hover:bg-space-cyan/10 transition-colors">
          {task.task_type_str}
        </span>
        <span className="px-3 py-1 text-sm rounded-full bg-space-dark/80 text-space-cyan border border-space-cyan/30 
        backdrop-blur-sm">
          User: {task.user_id ?? 'N/A'}
        </span>
        {getTaskTypeBadges().map((badge, index) => (
          <span
            key={index}
            className="px-3 py-1 text-sm rounded-full bg-space-dark/80 text-space-cyan border border-space-cyan/30 
            backdrop-blur-sm group-hover:border-space-cyan/50 transition-colors"
          >
            {badge}
          </span>
        ))}
        {isCompleted && (
          <span className="px-3 py-1 text-sm rounded-full bg-green-900/30 text-green-400 border border-green-500/30 
          backdrop-blur-sm">
            Completed
          </span>
        )}
      </div>

      {/* Details Section */}
      <div className="space-y-2 text-gray-300 font-mono text-sm backdrop-blur-sm">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          <p className="flex items-center justify-between bg-space-darker/50 rounded p-2 
          hover:bg-space-darker/70 transition-colors group/item">
            <span className="text-space-cyan font-semibold">Priority:</span>
            <span className="font-light tracking-wider">{task.priority?.toFixed(2) ?? 'N/A'}</span>
          </p>
          <p className="flex items-center justify-between bg-space-darker/50 rounded p-2 
          hover:bg-space-darker/70 transition-colors group/item">
            <span className="text-space-cyan font-semibold">Location:</span>
            <span className="font-light tracking-wider truncate">
              {task.latitude?.toFixed(2) ?? 'N/A'}°, {task.longitude?.toFixed(2) ?? 'N/A'}°
            </span>
          </p>
          <p className="flex items-center justify-between bg-space-darker/50 rounded p-2 
          hover:bg-space-darker/70 transition-colors group/item">
            <span className="text-space-cyan font-semibold">Duration:</span>
            <span className="font-light tracking-wider">{task.task_duration?.toFixed(2) ?? 'N/A'}s</span>
          </p>
          <p className="flex items-center justify-between bg-space-darker/50 rounded p-2 
          hover:bg-space-darker/70 transition-colors group/item">
            <span className="text-space-cyan font-semibold">Storage:</span>
            <span className="font-light tracking-wider">{task.storage_size?.toFixed(2) ?? 'N/A'}</span>
          </p>
          <p className="flex items-center justify-between bg-space-darker/50 rounded p-2 
          hover:bg-space-darker/70 transition-colors group/item">
            <span className="text-space-cyan font-semibold">Reward:</span>
            <span className="font-light tracking-wider">{task.task_reward?.toFixed(2) ?? 'N/A'}</span>
          </p>
          <p className="flex items-center justify-between bg-space-darker/50 rounded p-2 
          hover:bg-space-darker/70 transition-colors group/item">
            <span className="text-space-cyan font-semibold">Min Elevation:</span>
            <span className="font-light tracking-wider">{(task.min_elev ? (task.min_elev * 180 / Math.PI).toFixed(2) : 'N/A')}°</span>
          </p>
        </div>
        
        {task.sats_collecting?.length > 0 && (
          <p className="mt-2 bg-space-darker/50 rounded p-2 hover:bg-space-darker/70 transition-colors">
            <span className="text-space-cyan font-semibold">Collecting Sats:</span>{' '}
            <span className="font-light tracking-wider break-words">
              {task.sats_collecting.join(', ')}
            </span>
          </p>
        )}
      </div>
    </div>
  );
};

export default TaskCard;