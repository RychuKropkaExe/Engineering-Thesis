# Neural network inputs:
# map = MAPS_SET[mapName]
# x = state.position.x
# y = state.position.y
# z = state.position.z
# speed = state.display_speed
# velocity = state.velocity
# ypw = state.yaw_pitch_roll
# yaw = ypw[0]
# pitch = ypw[1]
# roll = ypw[2]
# turningRate = state.scene_mobil.turning_rate
# gerabox = state.scene_mobil.engine.gear
# currBlock = checkPosition(state.position)
# currBlockRotation = 
# nextBlocks = checkNextElements(state.position)
# nextBlock = nextElements[0]
# nextBlockRotation = 
# secondNextBlock = nextElements[1]
# secondNextBlockRotation = 
# distanceToCurrentBlock ? 
# distanceToNextBlock -> Distance to current block end
# distance to secondNextBlock -> Distance to next block end 
# iface.set_input_state(accelerate=True)