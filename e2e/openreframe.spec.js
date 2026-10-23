
const { test, expect } = require('@playwright/test');
const path = require('path');

test('OpenReframe upload, preview, and smart planning', async ({ page }) => {
  const imagePath = process.env.OPENREFRAME_TEST_IMAGE;

  expect(
    imagePath,
    'Set OPENREFRAME_TEST_IMAGE to the full path of a real JPG or PNG image.'
  ).toBeTruthy();

  // Open the running application.
  await page.goto('http://127.0.0.1:8000/');

  await expect(page.locator('#health')).not.toHaveText('server offline');

  // Upload a real image through the file input.
  await page.locator('#file').setInputFiles(path.resolve(imagePath));

 // Confirm the uploaded source image has loaded.
await expect.poll(async () => {
  return page.locator('#iOrig').evaluate(img => img.naturalWidth);
}).toBeGreaterThan(0);

  await expect(page.locator('#controls')).toBeVisible();
  await expect(page.locator('#actions')).toBeVisible();

  // Wait for the asynchronous geometry preview to finish.
  await page.waitForFunction(() => {
    const image = document.querySelector('#iPrev');
    return image && image.complete && image.naturalWidth > 0;
  }, null, { timeout: 180000 });

  await expect(page.locator('#cPrev')).not.toBeEmpty();

  // Switch to Smart Reframe. The UI automatically requests a plan.
  await page.locator('#mSmart').click();
  await expect(page.locator('#smartPanel')).toBeVisible();

  // Wait for candidates rather than assuming planning succeeded.
  await expect(page.locator('#candList .cand').first())
    .toBeVisible({ timeout: 180000 });

  await expect(page.locator('#genVars')).toBeEnabled();

  console.log('PASS: upload, geometry preview, and smart planning completed.');
});
