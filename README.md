# Frontend Service for IEEE Northwestern Competition 2025

## Authors
- **Mercy Muiruri**
- **Ayush Arora**
- **Emran Majidy**

## Background
My team and I won the annual Institute of Electrical and Electronic Engineers invite-only competition held in Northwestern, Chicago.

Many students found it hard to find places to volunteer around Evanston, our local town, and especially places that matched their interests.
So my team and I built NUVolunteers.org, an application that uses ML models to match students to volunteer opportunities that match their user profile and interests.

This is the frontend service of the system.

## Languages/Tools/Frameworks
- React.js      – 17 Core React components with the framework for building the UI. Components include LandingPage.js, Home.js, StudentDashboard.js, StudentProfile.js etc.
- TypeScript    – Provides static type safety and improves maintainability.
- HTML          – Form elements, labels, and ARIA-friendly components for accessible navigation.
- React Router  – Handles nested and protected routes for authenticated sections in my ProtectedRoute.js
- CSS           – Custom styling for  Student dashboard and Posts.js to display search feeds for users.
- Node.js       – Dependency management and development environment setup.

## Frontend Architecture Practices
- Security                    - Authenticated pages (/home, /dashboard, /profile) are secured with ProtectedRoute
- Flexbox & Grid              – Responsive layout design and adaptive component arrangement.
- Keyframe Animations         – Interactive button hover and bounce effects [(.bouncy-button](https://github.com/merc-cyy/raiso-ieee-2025/blob/main/frontend/tbd/src/App.css)) enhance UI 
- Responsive Design           – Media queries ensure usability across mobile, tablet, and desktop devices.
- Hooks (useState, useEffect) – Manages state and side effects in functional components.
- Separation of Concerns      – Logic, styling, and layout are modularized for scalability and maintainability.
- Performance & Optimization  – Lazy loading, modular component design, minimal re-rendering


## Acknowledgements
Many thanks to my team members:
- Ayush Arora
- Emran Majidy
- Breanna Lu
- Naomi Li
- Casey Zhang
for their great skills and camaraderie.

