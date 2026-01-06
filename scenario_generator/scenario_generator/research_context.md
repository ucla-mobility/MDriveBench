Driving is inherently a multi-agent task: drivers constantly negotiate gaps, turns, and priorities with others. Human drivers use implicit signals (small accelerations, eye contact, body language) to coordinate without explicit communication. For example, Mavrogiannis et al. (2022) model such interactions at uncontrolled intersections by abstracting multi-vehicle motions into topological braids. They show that, without explicit communication, agents can still coordinate by predicting interaction “modes” (who will pass first), greatly reducing collisions
personalrobotics.cs.washington.edu
. Similarly, game-theoretic models treat each vehicle as a player choosing actions (go or wait) and find Nash or Stackelberg equilibria for who yields. In a two-car intersection, a repeated Nash game often suffices, but Stackelberg (leader-follower) solutions – where the first-arriving car leads and the other yields – yield higher overall efficiency
arxiv.org
. These studies emphasize that implicit coordination (e.g. driving cues, hand-waves, inching forward) is critical, since “the lack of explicit communication… results in high uncertainty which complicates decision making”
personalrobotics.cs.washington.edu
.

Game-theoretic research on driving covers merges, lane changes, and intersections. For highway merging, researchers explicitly model the “merge-or-yield” game: each driver chooses to accelerate (join) or decelerate (give space). Studies report that modeling this negotiation as a Stackelberg (leader–follower) or Nash game can improve safety and efficiency
arxiv.org
. For example, Wei et al. (2022) use a Stackelberg differential game with MPC to plan safe merges
arxiv.org
, and Li et al. (2024) learn hidden “utilities” in human merging games via IRL
arxiv.org
. In general, game-based planners capture interactions: if both vehicles proceed at once, collision is high-cost, but if one yields, both make progress. Jia et al. (2022) propose a Nash-bargaining intersection model to mimic human courtesy: the solution often has one driver yielding “human-like” to avoid deadlock
arxiv.org
. At roundabouts (though not supported in our pipeline), similar game reasoning shows that a Stackelberg approach (first car goes, others follow) can achieve near-cooperative efficiency with decentralized decisions
arxiv.org
.

Another line of research highlights uncontrolled intersections and human behavior. Li et al. (2019) model traffic at unsignalized intersections with a game framework, reproducing real-world conflict resolutions
ar5iv.org
ar5iv.org
. They note that over one-fourth of U.S. fatal crashes occur at intersections, and many at uncontrolled ones
ar5iv.org
, underscoring the high stakes of these interactions. Crucially, without traffic signals or stop signs, “drivers need to decide when and how to enter…on their own” and failures to do so can cause deadlocks (if too conservative) or collisions (if too aggressive)
ar5iv.org
. In other words, implicit negotiation is the only guide. Pilot studies confirm that human drivers resolve ambiguous right-of-way using combinations of yield-to-the-right rules, courtesy gestures, or micro-movements – practices that are hard to encode in AV logic. Recent work by Tiwari et al. (2025) explicitly emphasizes “ambiguous right-of-way rules, occlusions, and unpredictable driver behavior” as major crash factors at uncontrolled intersections
arxiv.org
.

Long-Tail Scenario Taxonomy

“Long-tail” driving scenarios are rare, complex situations that fall outside the common training or everyday events, yet are crucial for safety. Hallgarten et al. (2024) stress that AV planners are usually tested on typical datasets (straight roads, common traffic) and may fail on rare combinations of actions
arxiv.org
. In a newly proposed “interPlan” benchmark, they include edge cases (e.g. unusual multi-vehicle conflicts, blocked roads) and show that neither rule-based nor learned planners handle them reliably
arxiv.org
. Long-tail scenarios often involve multiple unusual factors at once: for instance, many vehicles arriving simultaneously at an uncontrolled 4-way intersection, or a vehicle on the shoulder suddenly rejoining traffic as others turn.

We can categorize long-tail multi-vehicle situations as follows:

Unusual intersection conflicts: e.g. three or four vehicles arriving at once from different directions, each with different turn maneuvers (two left-turners vs. one going straight vs. one turning right). These can confuse right-of-way conventions.

Merge and lane-change congestion: multiple cars from on-ramps or shoulder lanes vying to enter or switch lanes in dense traffic. If two vehicles time their merges identically, it creates a complex implicit negotiation for a single gap.

Vulnerable road user interactions: adding pedestrians/cyclists crossing amid multi-car interactions. A pedestrian stepping into an intersection with multiple approaching cars is rare but critical.

Occlusion-driven conflicts: parked vehicles or props block sightlines, causing sudden emergence of a hidden vehicle. E.g. two cars start to cross, unaware of a third behind a truck.

Chain-reaction sequences: one driver hesitates or stops, causing a ripple of braking behind them. For example, a lead car yields and a platoon forms, then a new arrival behind the platoon cannot see the original conflict until late.

Mixed topologies: interactions spanning multiple segments (e.g. a merge followed immediately by a turn), or non-standard layouts (T-junctions with heavy lateral flow).

These long-tail cases share that they force non-obvious priorities and often lack clear right-of-way. They are not covered by simple signal patterns or traffic rules, making them “rare” in standard datasets but essential for robust AV testing
arxiv.org
arxiv.org
.

Coordination Challenge Patterns

Below we outline canonical conflict patterns, each described by physical setup, why it demands implicit coordination, and how it maps to our constraint language. Each pattern creates ambiguity in who should proceed:

Simultaneous Arrival (Head-on and Perpendicular): Setup: Vehicles 1 and 2 approach a 4-way intersection from opposite directions (Vehicle1 opposite_approach_of Vehicle2), both planning left turns (maneuvers left). Challenge: Left-turners arriving together must cross paths; standard “yield-to-right” or first-arrival rules give no clear priority. Constraints Used: opposite_approach_of, both left; possibly same_exit_as if they intend to turn to the same road. Why Ambiguous: Each might think the other will yield or go first; no signals break the symmetry. Variations: Change one to straight or right, or add Vehicle3 from a perpendicular direction (perpendicular_left_of/perpendicular_right_of) wanting another maneuver.

Merge Competition: Setup: On a multi-lane corridor, Vehicle1 (e.g. from an on-ramp) wants to merge into Vehicle2’s lane. Meanwhile Vehicle2 has a follower (Vehicle3) in the same lane (follow_route_of). Challenge: Vehicle1 and Vehicle2 vie for the same space: Vehicle1 merges_into_lane_of Vehicle2, while Vehicle3 (follow_route_of Vehicle2) cannot proceed until Vehicle2 yields. Constraints: merges_into_lane_of, follow_route_of. Ambiguity: Vehicle1 can’t signal, and Vehicle2’s decision affects Vehicle3. Without communication, each must guess if the other will slow or speed up. Variations: Multiple merging cars, or Vehicle2 simultaneously lane_change or left_lane_of another car, compounding the gap negotiation.

Opposing Left-Turners: Setup: Vehicles 1 and 2 approach from opposite directions (opposite_approach_of), both intending to turn left (left). Challenge: Each will cross into the other’s lane after the turn, intersecting at the center. Constraints: opposite_approach_of, both left, likely same_road_as after turning (if they end up on the same street). Ambiguity: Normally, they could turn simultaneously if they yield to oncoming straight traffic, but if both arrived together, they must implicitly decide who turns first. Variation: Add Vehicle3 approaching perpendicularly (perpendicular_right_of Vehicle1) also turning left. Now three-way negotiation ensues.

Narrow Passage: Setup: A two-lane corridor narrows to one lane (e.g. construction zone or a bridge). Vehicle1 and Vehicle2 approach from opposite ends. Constraints: We could model this as Vehicle1 opposite_approach_of Vehicle2, both aiming to straight. The bottleneck acts like an intersection. Ambiguity: Only one can pass at a time, but who? No signals exist; each driver must guess if the other will stop or go. Variation: Place a static_prop (barrier) narrowing the road. Or have Vehicle1 in a left_lane_of Vehicle2 (two-lane road), where Vehicle1 could pass Vehicle2; but if a sudden merge (merges_into_lane_of) is triggered, a conflict arises.

Platoon Chain Reaction: Setup: Vehicle1, Vehicle2, and Vehicle3 are in a queue (follow_route_of). Ahead, Vehicle4 is at an intersection deciding to turn. Vehicle4’s action (go or yield) indirectly affects all three. Constraints: follow_route_of(Vehicle2,Vehicle1), follow_route_of(Vehicle3,Vehicle2). If Vehicle4 same_approach_as Vehicle1, its decision propagates down the line. Ambiguity: Even if Vehicle4 defers to Vehicle1, Vehicle3 can’t see Vehicle4’s action until Vehicle2 moves. Implicitly, they must infer from minimal motion cues. Variation: Make Vehicle4 from a different road (perpendicular_left_of Vehicle1) causing a chain when it yields or stops.

Occluded Turn: Setup: Vehicle1 is at a T-junction (or 4-way) about to go straight. On the cross street, Vehicle2 (perpendicular approach) also goes straight. However, Vehicle3 is hidden behind a parked_vehicle at the corner, also approaching. Constraints: perpendicular_right_of(Vehicle2,Vehicle1). Vehicle3’s approach might be hidden until late. Ambiguity: Vehicle1 and Vehicle2 proceed as if only each other exist, but suddenly Vehicle3 appears. Neither expects it, testing their replanning ability. Variation: Replace parked_vehicle with a large static_prop or building edge, and/or have a cyclist cross in view.

Cross-Traffic with Vulnerable User: Setup: Vehicles 1 and 2 approach from opposite perpendicular directions, planning straight. A walker starts crossing the intersection perpendicularly (cross_perpendicular). Constraints: Non-ego actor walker (not listed in inter-vehicle constraints). Position walker “crossing.” Vehicles must implicitly decide to stop or proceed. Ambiguity: Neither car has explicit signal to the other; each may assume the other yields to the pedestrian. Variation: Use a cyclist instead of a walker, or have one vehicle turn right (so its path crosses the walker’s).

Each pattern above must be expressed using only the allowed constraints (no speeds or explicit yields). For example, “Vehicle1 opposite_approach_of Vehicle2” or “Vehicle1 merges_into_lane_of Vehicle2” are valid constraint phrases. These templates create genuine ambiguity: no rule or signal tells the vehicles what to do, so an implicit negotiation (who goes first) is needed.

Conflict Complexity Dimensions

Scenarios vary in difficulty along several axes:

Number of Vehicles: More vehicles exponentially increase possible interactions (e.g. three-car vs. four-car triangle vs. five-car star). Small groups (2–3) yield few simple modes; large groups (5–6) admit many modes (permutations of passing orders).

Approach Directions: Conflicts with vehicles from two directions are simpler (left–right or head-on), but three- or four-way arrivals greatly complicate who has priority. A 4-way tie often has no clear yield rule
arxiv.org
.

Maneuver Mix: Straight-on traffic is generally predictable, but mixing turn and straight maneuvers multiplies possibilities. (E.g. “straight vs. left vs. right” yields more ambiguous combinations than “straight vs. straight.”) Vehicles turning left often must yield to opposing straight traffic, but if no straight car exists, two left-turners face a symmetrical game.

Vulnerable Road Users (VRUs): Adding pedestrians or cyclists adds urgency and extra crossings. AVs must handle VRUs as dynamic obstacles without explicit right-of-way, elevating risk. Occlusions by pedestrians (around corners) can surprise drivers.

Visibility/Occlusion: Reduced sight (low light, obstructions) means drivers cannot see others until late. For instance, a car emerging from behind a building enforces a sudden decision. Studies note occlusions as a key crash factor at uncontrolled intersections
arxiv.org
.

Competing Intentions: When multiple vehicles share the same goal (e.g. two want the same merge gap, or three want to turn onto the same road), conflicts intensify. “Follow_route_of” platoons behind a leader can complicate chain reactions if the leader changes course unexpectedly.

Queue Depth and Pressure: More vehicles queued behind (followers) increase pressure to move forward. A deep queue can cause a leader to behave differently (e.g. blocking the cross street longer). Although not extensively studied, drivers with a platoon behind are often more cautious or feel a duty to yield (a social payoff), affecting the implicit game.

These factors interplay. For example, many vehicles (high number) from three directions (high approach count) with mixed turns and a pedestrian crossing, all under occlusion, would be extremely challenging. By contrast, two cars from opposite directions both going straight is trivial. Importantly, any scenario with ambiguous right-of-way – whether due to symmetry, lack of signals, or competing maneuvers – forces the agents to implicitly negotiate via motion cues.

What AVs Struggle With in Multi-Agent Scenarios

Modern AV systems excel in structured environments, but multi-agent interactions expose their limitations. A key weakness is prediction uncertainty: AV planners often assume “typical” human behavior, but in rare jams or courtesy situations humans act unpredictably. When multiple cars interact without clear rules, the planner may oscillate between being too cautious (deadlocks) or too aggressive (near misses). Hallgarten et al. showed that neither rule-based nor learned planners could safely navigate complex long-tail scenarios
arxiv.org
, implying many AVs lack robustness to unusual interactions.

Sensor and perception issues also loom large. Occluded objects (vehicles or pedestrians) can appear suddenly, and if the AV’s sensors or tracking falter, the system may not react in time. The importance of occlusion is underscored by Tiwari et al.: they list occlusions and unpredictable behavior as top causes of intersection crashes
arxiv.org
. Game-theoretic planners assume other agents are somewhat rational; in reality, drivers may be distracted or erratic, breaking those models. AVs have no “common sense” shortcuts like eye contact or hand waves.

In short, AV failures in multi-vehicle scenarios often stem from underestimating the negotiation problem. For example, an AV might not correctly infer who will yield if two vehicles inch forward in sync, leading to dangerous stalemates or near-collisions. Historically, many AV disengagements have occurred at intersections or merging lanes, suggesting prediction/planning limits when facing concurrent agents. Factors cited include sensor blind spots, misclassification of agent intent, and planning algorithms that optimize selfishly (time or comfort) without accounting for social cues
arxiv.org
arxiv.org
.

Guidance for Scenario Generation

To create diverse, challenging multi-agent scenarios within our constraint system:

Combine Multiple Constraints: Stack relational constraints to form rich interactions. For instance, combine opposite_approach_of(V1,V2), left(V1), straight(V2), and also same_approach_as(V3,V1) if adding a third in the same lane. Each added relation (e.g. merges_into_lane_of, follow_route_of) introduces new dependencies.

Scale Difficulty Gradually: Start with 3 vehicles and then increase. More vehicles and more approach directions make priority less obvious. Add pedestrians or cyclists (walker, cyclist with cross_perpendicular) to stress-test AV perception. Narrow passages and parked vehicles test spatial constraints.

Enforce Implicit Priority: Never use explicit yield or go ordering. Don’t write “Vehicle1 yields to Vehicle2”. Instead, rely on timing hints: place vehicles “on_approach” simultaneously. For example, describe “Vehicle1 and Vehicle2 arrive at the intersection together; both must decide independently who goes first.” The LLM must not label one as leader.

Mix Maneuvers Strategically: A good conflict combines different maneuvers: straight vs. left, or two left-turners from different roads, or a lane-change across an oncoming path. For example, one pattern is a “right-turn intercept” where Vehicle1 is turning right (cutting across another lane) while Vehicle2 is going straight from the side. Use the right maneuver plus approach constraints to generate uncertainty.

Use Occlusions for Surprise: Introduce parked_vehicle or static_prop at junctions to hide an actor. This tests the AV’s ability to update its plan when a hidden VehicleX emerges. For instance: “Vehicle3 is occluded by a parked car at the corner; Vehicles1 and 2 proceed as if alone, then Vehicle3 appears at the intersection.”

Maintain Realism: Keep scenarios plausible. All vehicles should have legal maneuvers on given lanes. For example, don’t make a car turn left from a right-edge position (right_lane_of or offroad_right should match a straight-going car, not a left-turner).

Avoid Trivial Resolutions: Don’t create situations with an obvious “only one answer” or where the implicit coordination is essentially forced by constraints. The goal is genuine hesitation, not trivial compliance.

Quality Metrics: After generation, evaluate scenario difficulty using metrics from the literature. Time-to-collision (TTC), minimum distance, required deceleration, or predicted collision probability are useful indicators of challenge
preprints.org
. Scenarios with low TTC or high collision probability are high-stress tests. Measuring diversity (different combinations of constraints) ensures variety.

Failure-Exposing Cases: To expose AV prediction or planning failures, craft scenarios where plausible human-like actions lead to different outcomes. For example, two left-turners inch forward (ambiguous); or a platoon where the leader brakes unexpectedly. Such cases highlight any planner blind spots.

Common mistakes to avoid include implicitly giving one vehicle priority by, for example, placing it slightly “in the intersection” (in_intersection) earlier than others, or specifying timing offsets. All ego vehicles should start “on_approach” together unless modeling a true delay. Also avoid scenarios that require exact speed or time, since these are not part of the constraint system.

Finally, implicit coordination means the scenario description must never state or imply “who should go.” Instead, describe the geometry and maneuvers fully, then let the AV’s decision-making face the ambiguity. Effective scenarios leave even human observers wondering “Hmm, who would go first?” without a clear legal cue.

Long-Tail vs. Common Scenarios

A common driving scenario has a clear resolution (e.g. one car has obvious priority or signals exist), whereas a long-tail case nests multiple rare factors. For example, a single car arriving at an intersection alone is common; two cars from opposite directions both turning left at exactly the same time is rare. Research shows that standard AV tests cover mostly the former
arxiv.org
. In our taxonomy, long-tail scenarios often involve: more vehicles (3–6) than usual; no signals; mixed maneuvers; and potential occlusions – factors that together make resolution ambiguous. By contrast, removing any one of these (like having only 2 vehicles or visible traffic lights) would make the scene common and easy.

Answers to Specific Questions

Prevalence of Multi-Agent Coordination: Nearly all driving involves other road users, but the literature lacks an exact fraction of “genuine coordination” events. However, the high crash rates at intersections imply it is substantial: e.g. 24.5% of all fatal crashes were intersection-related in one U.S. study
personalrobotics.cs.washington.edu
. Similarly, highways routinely involve merge/lane-change games. Thus, a significant portion of driving (likely in urban areas 10–30%) requires multi-agent negotiation.

Common AV Failure Causes: Real-world AV failures often stem from perception and prediction errors in multi-agent contexts. For instance, the NHTSA cites that many automated driving accidents occur during lane changes or at crossings where the AV mispredicted a human’s action. In the multi-vehicle domain, key causes include: inadequate prediction of other vehicles’ intents; lack of planning for emergent behaviors (like a cut-in); and misinterpretation of human signals. Studies find that “rare, unpredictable behaviors” and interaction ambiguities are a leading cause of disengagement in mixed traffic
arxiv.org
.

Human Resolution of Ambiguities: Human drivers rely on default rules (e.g. yield-to-right), courtesy, and dynamic gap acceptance. If two drivers arrive simultaneously, many will yield to the right or gingerly inch forward to signal intent. Research (e.g. Tiwari et al.) indicates drivers use such implicit cues rather than formal negotiation
arxiv.org
. In cross-cultural terms, most Western countries teach “yield to the right in absence of signs”, but in practice drivers also use eye contact and subtle motion. (The pipeline does not model eye contact, so scenarios should focus on positional cues.)

Intersection Conflict Difficulty Factors: Intersections are harder when right-of-way is unclear (e.g. no dominant road, arrival at same time), sightlines are blocked, or conflicting turns are involved. Ambiguities grow with the number of active approaches and when standard rules (like “straight goes before left”) do not resolve all conflicts. For example, two left-turners from opposite ends with no others present have no clear priority under simple rules – making it quite difficult to resolve safely. Studies note ambiguous rules plus occlusions (“two factors”) account for many intersection crashes
arxiv.org
.

Queue Depth Effects: When vehicles queue up, the leader’s decision heavily influences the followers. A long queue can pressure the front vehicle either to take a gap (letting the tail go) or to block traffic (forcing all to wait). This can create multi-stage negotiation: followers must guess the leader’s intention. For example, if Vehicle1 in a queue signals (implicitly) to yield, Vehicles2–3 behind also stop, possibly causing confusion to cross traffic. There is evidence that dense platoons make negotiation slower, but detailed studies are scarce.

Role of Occlusion: Occlusions dramatically increase coordination failure risk. If a driver cannot see an oncoming car or pedestrian until the last moment, the situation turns into a surprise stop-or-go problem. By hiding one agent, occlusions transform a predictable interaction into a long-tail case. Research explicitly lists occlusion as a top factor in multi-agent crashes
arxiv.org
. In scenario terms, adding a parked_vehicle or building corner that hides an ego vehicle until “in_intersection” pushes the problem into the rare category.

Cultural/Regional Differences: Traffic rules vary. In many countries (U.S., Canada, Europe), default protocols at uncontrolled intersections are “yield-to-the-right” or “first-to-arrive”. In others, courtesy and informal negotiation play a bigger role. For instance, studies show more aggressive, less rule-bound driving styles in some regions (e.g. drivers in Southern Europe or parts of Asia may rely on assertiveness). Our constraint-based description should avoid implying any specific law; instead, emphasize the need for negotiation.

Game-Theoretic Equilibria: In common driving interactions, equilibria often manifest as one driver acting first while others yield. A static Nash equilibrium might be both drivers hesitant (mutual yield – a deadlock, rarely stable in practice) or both go (collision, of course avoided). Realistic solutions resemble Stackelberg: the “leader” goes first and the “follower” yields, often reflecting who arrived first or was slightly more aggressive. For example, game models of unsignalized conflict typically use Nash/Stackelberg equilibria where one car proceeds (leader) and others yield, balancing safety vs. progress
arxiv.org
arxiv.org
.

Exposing Prediction Failures: Scenarios that break prediction models often involve rare alignments of behavior. For example, an extreme cut-in (Vehicle rapidly merging across multiple lanes ahead of another), or a rapid sequence of moves (car brakes hard then swerves) are unlikely in training data. Designing scenarios where an ego vehicle must replan on-the-fly – such as a new vehicle entering the scene late – can reveal prediction weaknesses. The “long-tail” benchmark [25] specifically includes such edge cases to test planners.

Long-Tail vs. Common: A scenario is common/easy if simple traffic conventions or laws resolve it (e.g. one vehicle is clearly on the major road). It’s long-tail/hard if it stacks rare elements: multiple similarly-situated vehicles, equal priority, occlusion, unusual road geometry, or simultaneous conflict with VRUs. In practice, if a human driver would say “This is tricky, I’m not sure who should go,” the scenario is long-tail. Conversely, if a quick glance reveals one car on the main road or an obvious arrow, it is common.