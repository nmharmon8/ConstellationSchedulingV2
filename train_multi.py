from bsk_rl import sats, act, obs, scene, data, comm
from bsk_rl.sim import dyn, fsw

class ImagingSatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.OpportunityProperties(
            dict(prop="priority"), 
            dict(prop="opportunity_open", norm=5700.0),
            n_ahead_observe=10,
        )
    ]
    action_spec = [act.Image(n_ahead_image=10)]
    dyn_type = dyn.FullFeaturedDynModel
    fsw_type = fsw.SteeringImagerFSWModel