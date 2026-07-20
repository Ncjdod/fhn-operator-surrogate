"""
    HHSurrogate

GPU-batched Hodgkin–Huxley simulation, a control-affine flow-map surrogate with a closed-form
inverse for real-time control, a multi-compartment cable + line-source electrical image (EI),
and a differentiable biophysical inverse (parameter recovery + neurostimulation design).

Julia port and Hodgkin–Huxley extension of the `fhn-operator-surrogate` project, taking reference
from Lotlikar et al. (2026), *Learning Biophysical Models of Large-Scale Multineuronal Data to
Enable Precise Neurostimulation*.

The whole package runs and is unit-tested on the CPU with only StaticArrays + KernelAbstractions
installed; adding CUDA and passing `CuArray`s moves the batched work to the GPU with no code
change (KernelAbstractions picks the backend from the array type). See the README for the native
Windows + GTX 1660 Ti (no WSL2) setup.
"""
module HHSurrogate

# ---- core numerics (dependency order) ----
include("models/models.jl")        # HHClassic, MultiChan, RGCChannels; vfield, rest_state, ...
include("ad.jl")                    # in-house forward-mode Dual (no binary-artifact deps)
include("solvers/solvers.jl")      # batched RK4 + Rosenbrock (ROS2) KernelAbstractions kernels
include("models/cable.jl")         # multi-compartment cable (implicit axial IMEX solver)

# ---- surrogate + control ----
include("surrogate/nn.jl")         # dense MLP toolkit + Adam (explicit fwd/bwd, GPU-batched)
include("surrogate/affine.jl")     # control-affine flow-map F(x)+G(x)u, closed-form inverse
include("surrogate/train.jl")      # curriculum BPTT training
include("data.jl")                 # coarse-grid ZOH operator dataset
include("control/inverse.jl")      # closed-form / lin1 / Gauss-Newton controllers, closed loop

# ---- extracellular + differentiable inverse (article) ----
include("inference/extracellular.jl")  # line-source EI + differentiable EI features
include("inference/fit.jl")            # gradient-based parameter recovery + stimulus design

# models & interface
export NeuronModel, HHClassic, MultiChan, RGCChannels
export vfield, rest_state, random_init, statedim, u_bounds, firing_band, gate_inf
# AD
export Dual, value, partials, seed, extract_jacobian
# solvers
export rollout_rk4, rollout_rosenbrock, rk4_substep, rk4_coarse, ros2_step
# cable + EI
export HHCableChannels, CableGeometry, CableState, straight_axon_geometry, cable_rest,
       simulate_cable
export hex_electrode_patch, electrical_image, ei_features
# surrogate + control
export AffineFlowMap, FG, flow_step, invert, rollout, params, loss_and_grads, to_device!
export train!, standardize_stats, Adam, adam_step!, clip_global_norm!, global_norm
export make_dataset, control_sensitivity
export control_lin1, control_gn, closed_loop, phi_and_sens
# differentiable inverse
export fit_conductances, spike_probability, stimulus_threshold, design_stimulus

end # module
