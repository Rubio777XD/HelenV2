# Alarm Manager Regression Checklist

These manual steps exercise the cross-page alarm behaviour introduced by the
shared alarm manager.

1. Start the HELEN backend and frontend as usual.
2. From the Home screen schedule an alarm for the next minute.
3. Navigate away to another section (for example **Configuración** or
   **Cronómetro**) without refreshing the browser window.
4. When the alarm expires verify that:
   - The global popup is shown on top of the current page.
   - The alarm tone plays immediately (unlock the audio once if the browser
     requires it).
5. Dismiss the alarm using the **Detener** button and repeat the flow from a
   different page to confirm consistency.

Record the outcome in the QA log to keep track of the scenario.
