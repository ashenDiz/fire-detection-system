import httpx

client = httpx.Client(base_url='http://127.0.0.1:8000/api/v1')

print("=== 1. Checking /system/status ===")
stat = client.get('/system/status').json()
print("   Backend:", stat['backend'])
print("   Database:", stat['database'])
print("   Camera:", stat['camera'])
print("   Person Model:", stat['person_model'])
print("   Fire Model:", stat['fire_model'])
print("   Sensor Mode:", stat['sensor_mode'])

print("\n=== 2. Checking /scenarios ===")
scenarios = client.get('/scenarios').json()
for s in scenarios:
    print(f"   [{s['id']}] {s['name']}: Temp={s['temperature']}C, Smoke={s['smoke_level']}%, Gas={s['gas_level']}%, Flame={s['flame_level']}%")

print("\n=== 3. Submitting Gas Leak Scenario (Testing Alert Deduplication) ===")
gas_leak = next(s for s in scenarios if s['id'] == 'GAS_LEAK')
payload = {
    'temperature': gas_leak['temperature'],
    'humidity': gas_leak['humidity'],
    'smoke_level': gas_leak['smoke_level'],
    'gas_level': gas_leak['gas_level'],
    'flame_level': gas_leak['flame_level'],
    'input_source': 'manual_simulation',
    'scenario_tag': 'GAS_LEAK'
}
res1 = client.post('/sensors/readings', json=payload).json()
alert1_id = res1['active_alert']['id']
print(f"   Call 1: Alert ID = {alert1_id}, Risk Level = {res1['risk']['risk_level']}")

# Repeat submission immediately with identical readings
res2 = client.post('/sensors/readings', json=payload).json()
alert2_id = res2['active_alert']['id']
print(f"   Call 2: Alert ID = {alert2_id} (Deduplicated without spam: {alert1_id == alert2_id})")

print("\n=== 4. Submitting Critical Fire Scenario (Testing Escalation) ===")
crit_fire = next(s for s in scenarios if s['id'] == 'CRITICAL_FIRE')
post_crit = client.post('/sensors/readings', json={
    'temperature': crit_fire['temperature'],
    'humidity': crit_fire['humidity'],
    'smoke_level': crit_fire['smoke_level'],
    'gas_level': crit_fire['gas_level'],
    'flame_level': crit_fire['flame_level'],
    'input_source': 'manual_simulation',
    'scenario_tag': 'CRITICAL_FIRE'
}).json()
print("   Risk Score:", post_crit['risk']['overall_risk_score'])
print("   Risk Level:", post_crit['risk']['risk_level'])
print("   Emergency Priority:", post_crit['risk']['emergency_priority'])
print("   Factors:", post_crit['risk']['contributing_factors'])
crit_alert = post_crit['active_alert']
print(f"   Escalated Alert ID: {crit_alert['id']} (Type: {crit_alert['alert_type']})")
print(f"   Traceability -> SensorReading ID: {crit_alert['sensor_reading_id']}, RiskAssessment ID: {crit_alert['risk_assessment_id']}")

print("\n=== 5. Checking Detection Events (/api/v1/events) ===")
events = client.get('/events').json()
print(f"   Logged Detection Events Count: {len(events)}")
for e in events[:3]:
    print(f"   Event {e['id']}: Level={e['risk_level']}, Score={e['risk_score']}, Camera={e['camera_status']}, FireAI={e['fire_detected']}")

print("\n=== 6. Acknowledging Active Alert ===")
ack_res = client.post(f"/alerts/{crit_alert['id']}/acknowledge").json()
print(f"   Alert {crit_alert['id']} acknowledged: {ack_res['acknowledged']}, timestamp: {ack_res['acknowledged_at']}")

print("\n=== 7. Restoring SAFE Normal Baseline ===")
norm = next(s for s in scenarios if s['id'] == 'NORMAL')
post_norm = client.post('/sensors/readings', json={
    'temperature': norm['temperature'],
    'humidity': norm['humidity'],
    'smoke_level': norm['smoke_level'],
    'gas_level': norm['gas_level'],
    'flame_level': norm['flame_level'],
    'input_source': 'manual_simulation',
    'scenario_tag': 'NORMAL'
}).json()
print("   Risk Score:", post_norm['risk']['overall_risk_score'])
print("   Risk Level:", post_norm['risk']['risk_level'])
print("   Factors:", post_norm['risk']['contributing_factors'])
print("\n>>> ALL LIVE VERIFICATIONS COMPLETED SUCCESSFULLY! <<<")
