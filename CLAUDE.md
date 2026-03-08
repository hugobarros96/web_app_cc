# Scheduling app
This is a python web app that solves an optimization problem. The app works so that the Scheduler user (PT) can fit all the scheduling users into his schedule. For instance for a professor to schedules classes with different students. Or a Personal trainer to schedule classes with different clients.

## App feature definition
Given several users, each with an available schedule and a Scheduler user (PT) that has to fit all the users availability in pre defined slots.
As example:
    - Scheduler user, from now on referred as PT PT schedule: available from 10h monday to 18h tuesday.
    - user 1: available from 10h to 11h on monday and 14h to 15:30h on tuesday (slots user 1 need to fill: 1h)
    - user 2: available from 10h to 11:30h on monday ((slots user 1 need to fill: 1h))
    - The user 2 should be set to its only available slot 10h to 11h while user 2 can be set to 14h to 15:30h on tuesday bacause he is also free there
- The PT as total control of the users it creates/deletes
- The PT can set its own available schedule (constrain to where the other slots can be fitted)
- The optimization problem should be fro the PT total available schedule.
- We should allow up to 50 normal users ( besides PT):
    - each user has a name and associated slots of time (1 slot min and 4 slots max)
        - Each slot has 30min min and 1:30h max
    - we can add or remove slots from users

# UI
- calendar UI display a calendar of the week
- on the left side there is a menu bar:
    - in the left bar we can see and select the users (each user has a random color)
        - drop down menu for user: each user also displays the number of slots associated with that user and the time of each slot. each slot is an item.
        - drop down menu for user: the slot has a "X" button that delets the slot form that user
        - drop down menu for user: a "add slot" button to add a slot to that user
    - we can also see the PT user or "Scheduler Availability" with Highlight in the same left bar
    - we can create a new user via a + button at the end - the user needs a name and their time slots defined upon creation
    - we can delete a user via a garbage button in from of the user name
    - When selecting a user we can draw in the calender the schedules available for that user. The spaces in the calender should have different colors - same as user color. I.e. for th PT we can select in the calendar all the available schedule.
- The availability of the Scheduler user (PT) is always shown in the calender (fixed color preferably more transparent).
- When we select any user we should see only the availability of that user and the one of the PT (translucid) - when we de-select all users all availabilities should be shown in then calender even if they overlap
- Button called "Start Scheduling" that will star the optimization code.
- Results are presented in a tab page and teh results should be in text as well as in graphic form in  the calender.

# Deployment ENV
- Docker image
- web app
- link should be hugobarros.cc/scheduler
- IP deployment VM: 35.231.149.237
- Domain: hugobarros.cc