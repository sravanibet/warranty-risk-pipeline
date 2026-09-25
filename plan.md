plan 



"I'm turning the pipeline into an agent. Four phases: it asks the user to describe the facility and confirms it visually, then takes in data — real if there is any, simulated only if there isn't — then does the learning, then scores vehicles and explains why. The agent orchestrates and explains; the Python modules do the actual computation. The simulator moves out of the app into its own optional package."


The Quality Monitoring Agent

The system has four phases, in the order you described.

1. Setup. The agent explains what it needs to know about a facility — stations, operations, sensors and their types, how stations connect, and how the product moves. The user gives it a configuration file. The agent validates it, reports every problem at once, and draws the line back as a diagram so the user can confirm it was understood correctly. Nothing proceeds until that confirmation.

The configuration format is general rather than tied to one automotive line, so it can describe a job shop, parallel machines, or a product that stays in place. Topology and routing are kept separate: what the facility is, versus the path a given product takes.

2. Data. The agent asks whether real data exists. If it does, it's uploaded and checked against the registered configuration — right sensors, right types, nothing missing. If there is no real data, the agent offers to simulate. The simulator lives in its own package outside the application, because a real plant would never use it.

3. Learning. Tolerance optimisation and model training run here, inside the agent, with no separate tabs. Labelling and preprocessing become internal steps rather than user-facing stages. The agent reports results in plain language and flags what looks wrong.

4. Operate. The agent scores each vehicle, explains what drove the score, and recommends an action. Station-level statistical control stays where it is — the agent reads its output rather than replacing it, and looks for the line-level and plant-level patterns instead.

Later, the data comes from MIX rather than an upload. The internal functions stay the same; only the source changes.