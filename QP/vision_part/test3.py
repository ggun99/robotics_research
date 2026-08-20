import rbpodo as rb

rb3 = rb.Cobot("192.160.0.10")
# rb3.activate()
rc = rb.ResponseCollector()

rb3.set_operation_mode(rc, rb.OperationMode.Real)
if rb3.get_robot_state(rc):
    print(rb3.get_robot_state(rc))

rb3_data = rb.CobotData("192.160.0.10")
rb3_data_js = rb3_data.request_data().sdata.jnt_ang
print(f"RB3 Current Joints: {rb3_data_js}")

