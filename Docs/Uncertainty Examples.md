# DDORWA Project

## The Travelling Salesman Problem — A Foundation

The **Travelling Salesman Problem (TSP)** is one of the most studied and celebrated problems in combinatorial optimization and computer science. In its classical form, the problem is deceptively simple to state: given a set of cities and the distances between every pair of them, find the shortest possible route that visits each city exactly once and returns to the starting point. Despite this straightforward description, the TSP belongs to the class of **NP-hard** problems, meaning that no polynomial-time algorithm is known to solve all instances optimally, and the search space grows factorially with the number of cities.

The TSP serves as a canonical benchmark for optimization techniques ranging from exact methods — such as branch-and-bound and dynamic programming — to metaheuristics like simulated annealing, genetic algorithms, ant colony optimization, and more recently, neural combinatorial optimization. Its practical relevance is equally broad: scheduling, circuit board drilling, genome sequencing, and — most prominently — **logistics and delivery routing** all reduce to or can be modeled as variants of the TSP.

In most textbook treatments the TSP lives in a clean, deterministic world: distances are fixed, every city is always reachable, and the traveler suffers no surprises along the way. Real-world routing, however, is anything but clean.

---

## Extending the TSP: Real-World Uncertainty in Last-Mile Delivery

This repository investigates an extended, uncertainty-aware formulation of the TSP that is directly motivated by the challenges encountered in **last-mile logistics and delivery operations**. The classical problem is enriched with a rich set of stochastic and dynamic factors that reflect the messy reality of operating a fleet of vehicles in an urban or peri-urban environment. The goal is to develop models, algorithms, and decision-support tools that remain robust and adaptive when the world refuses to cooperate with the plan.

Below is a detailed account of each uncertainty dimension addressed in this project.

---

### 1. Stochastic Travel Times

In the deterministic TSP every arc carries a fixed cost. In practice, the time it takes to traverse a road segment fluctuates continuously due to congestion, traffic signal timing, pedestrian crossings, and the sheer variability of urban traffic flow. These travel times are better represented as **random variables** — often dependent on the time of day, day of week, and local events — rather than fixed scalars. A routing plan that looks optimal under average conditions may perform poorly once actual travel-time realizations are taken into account. This project models travel-time distributions explicitly and seeks routes that are both expected-cost-efficient and robust against adverse realizations.

### 2. Road Closures and Accidents

Planned road works, emergency lane closures, and traffic accidents can instantaneously remove arcs from the network or inflate their traversal cost dramatically. Unlike routine congestion, these events are largely **unpredictable in advance** and may not be reflected in any pre-trip traffic data. A vehicle mid-route may discover that its planned path is blocked entirely, forcing costly detours and cascading delays across the rest of the schedule. This project considers dynamic arc-removal events and explores re-routing strategies that minimize the downstream impact on delivery commitments.

### 3. Weather-Related Disruptions

Adverse weather — heavy rain, snow, ice, dense fog, or extreme heat — degrades road conditions, reduces safe driving speeds, and in severe cases makes certain routes temporarily impassable. Weather disruptions are spatially and temporally correlated: a storm cell can simultaneously affect many arcs in a region, creating **correlated uncertainty** that is harder to hedge against than independent noise. This project incorporates weather forecasts and scenario-based reasoning to produce plans that remain feasible and near-optimal across a range of likely weather outcomes.

### 4. Vehicle Breakdowns

Mechanical failures, punctures, and fuel issues can render a vehicle temporarily or permanently unable to continue its assigned route. A breakdown mid-tour creates an urgent need to **reassign the remaining stops** to other vehicles in the fleet, potentially violating capacity constraints and delivery windows in the process. This project treats vehicle reliability as a probabilistic attribute and designs contingency re-routing protocols that can be activated in real time when a vehicle goes offline.

### 5. Service-Time Variability

The time a driver spends at each customer location — unloading goods, obtaining a signature, navigating a building's access procedures, or waiting for a customer to answer the door — is rarely the tidy constant assumed in classical models. Service times vary with the size and fragility of the shipment, the physical layout of the delivery site, and the customer's own responsiveness. Underestimating service time at early stops causes cascading lateness across all subsequent stops in the tour. This project treats stop-level service times as **random variables** and builds schedule buffers and dynamic re-sequencing logic to absorb the variability gracefully.

### 6. Time-Window Uncertainty

Many delivery contracts specify a **promised time window** within which the customer expects to receive their order. In practice both sides of that window are subject to uncertainty: the customer's availability may shift, and the driver's actual arrival time depends on all the stochastic factors described above. Failing to deliver within the agreed window typically incurs a **financial penalty** or necessitates a re-delivery attempt, both of which erode the economic viability of the route. This project models penalty functions for window violations and incorporates them into the objective, seeking plans that balance travel efficiency against the risk of incurring penalties.

### 7. Dynamic Customer Requests and Cancellations

Unlike the static TSP where the set of destinations is fixed before the tour begins, real delivery operations receive **new orders and cancellations throughout the working day**. A customer may place a same-day delivery request after the morning dispatch, or cancel an order while the driver is already in transit. The routing plan must therefore be treated as a **living document** that is updated continuously as new information arrives, balancing the cost of re-optimizing against the benefit of incorporating the latest demand picture.

### 8. Driver Availability, Leave, Strike, and Legal-Hours Constraints

The fleet's operational capacity is not a fixed resource. Drivers may call in sick, take planned leave, or participate in industrial action with varying degrees of forewarning. Even when a driver is available, **legal working-hour regulations** — such as mandatory rest breaks, maximum consecutive driving time, and shift-length caps — impose hard constraints on how long a route can run and when a driver must stop. This project models driver availability as a stochastic resource and embeds legal-hours compliance directly into the feasibility checks of every candidate route.

### 9. Parking and Address Search Uncertainty

Urban delivery is plagued by a class of delays that do not appear in any road network model: **finding a legal parking spot near the delivery address** and, separately, **physically locating the correct entrance** to a building or complex. Apartment blocks with obscure rear entrances, industrial estates where the address maps to a gatehouse rather than the relevant loading bay, or dense city centers with chronic parking scarcity can add substantial unplanned time to a stop. This project accounts for these last-fifty-meter uncertainties by augmenting stop-level service-time distributions and by allowing spatial clustering of stops to reduce aggregate parking search cost.

### 10. Capacity and Load-Related Constraints

Delivery vehicles have finite payload capacity in terms of weight and volume, and the **actual weight or volume of a shipment** may differ from its declared value — particularly for orders packed and sealed by customers or third-party warehouses. Cumulative load errors can cause a vehicle to reach capacity before completing its planned route, requiring an unscheduled return to the depot for reloading. This project models load uncertainty explicitly and incorporates probabilistic capacity constraints that ensure feasibility is maintained with high probability even when individual item sizes deviate from their nominal values.

### 11. Backlog of Deliveries and Inter-Day Carry-Over

Not every delivery attempt succeeds on the first try, and operational shocks — a vehicle breakdown, a sudden spike in same-day orders, or a weather closure — can leave a **backlog of undelivered items** at the end of the working day. These items must be folded into the next day's (or next period's) routing plan, creating dependencies that stretch across planning horizons. This project models the multi-period structure of the delivery problem, explicitly representing the carry-over of unfinished work and optimizing schedules that account for the accumulated backlog alongside fresh demand.

### 12. Customer Availability

A delivery is only successful if the customer is reachable at the time of arrival. Customers may be absent from home, unreachable by phone, or temporarily unavailable due to meetings or appointments, leading to **failed delivery attempts** that waste the driver's time and require the item to be returned to the vehicle or a local depot. This project treats customer availability as a time-dependent probability, uses historical contact-success patterns to inform scheduling decisions, and designs re-try policies that maximize the likelihood of first-attempt success.

### 13. Goods Returns, Re-Delivery, and Rescheduling

The delivery process does not always end with the handover of a parcel. Customers may **reject a delivery** because an item arrived damaged, failed to meet their expectations, or was simply ordered by mistake. In each case the driver must collect the return, log the reason, and — if a replacement is warranted — the item must be rescheduled for re-delivery at a future date. This creates a **reverse logistics loop** that interacts with the forward delivery schedule: the vehicle now carries a returned item that reduces its effective remaining capacity, and the depot must plan a new outbound delivery for the affected customer. This project models the stochastic occurrence of returns and integrates reverse-logistics handling into the overall routing and capacity management framework.

---

## Summary

Taken together, the factors above transform the clean combinatorial structure of the classical TSP into a rich, **stochastic and dynamic vehicle routing problem** that must be solved under partial and evolving information. The objective is not merely to find a short tour, but to find **adaptive policies** — decision rules that specify what to do as the real world reveals itself — that perform well in expectation, remain robust against worst-case scenarios, and can be re-optimized quickly enough to be useful in an operational setting.

This repository is the home of the models, algorithms, and empirical evaluations developed in pursuit of that goal.
